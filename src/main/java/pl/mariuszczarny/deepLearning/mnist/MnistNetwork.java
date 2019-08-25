package pl.mariuszczarny.deepLearning.mnist;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.beust.jcommander.Parameter;

public class MnistNetwork {
	private static final Logger log = LoggerFactory.getLogger(MnistNetwork.class);	
	private static final int workerPrefetchNumBatches = 2;
	private static final int averagingFrequency = 5;
	@Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
	private static int batchSizePerWorker = 32;
	@Parameter(names = "-numEpochs", description = "Number of epochs for training")
	private static int numEpochs = 4;
	private static String saveDir = "MyMultiLayerNetwork.zip";
	private static final boolean isSaveUpdater = true;
	private static final int seed = 12345;
	
	private static JavaRDD<DataSet> createRDD(JavaSparkContext sc, boolean isTrainingData) throws IOException {
		List<DataSet> dataList = new ArrayList<>();
		DataSetIterator dataSetIterator = new MnistDataSetIterator(batchSizePerWorker, isTrainingData, seed);
		
		while (dataSetIterator.hasNext()) {
			dataList.add(dataSetIterator.next());
		}
		
		return sc.parallelize(dataList);
	}

	public static void createNetwork(JavaSparkContext sparkContext, boolean isSaved) throws IOException {
		JavaRDD<DataSet> trainingData = createRDD(sparkContext, true);
		JavaRDD<DataSet> testData = createRDD(sparkContext, false);
		MultiLayerConfiguration networkConfig = NeuralNetworkConfig.getCnnNetwork(seed);
		TrainingMaster<?, ?> trainingConfig = configureTraining();
		SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sparkContext, networkConfig, trainingConfig);

		executeTraining(trainingData, sparkNetwork);
		Evaluation evaluation = performEvaluation(testData, sparkNetwork);

		if (isSaved) {
			saveResult(evaluation, sparkNetwork.getNetwork());
		}
		
		doCleaning(sparkContext, trainingConfig);
	}

	public static void restoreNetwork(JavaSparkContext sparkContext) throws IOException {
		JavaRDD<DataSet> trainingData = createRDD(sparkContext, true);
		JavaRDD<DataSet> testData = createRDD(sparkContext, false);
		
		SparkDl4jMultiLayer sparkNetwork = loadResult(sparkContext, configureTraining());
		
		executeTraining(trainingData, sparkNetwork);
		Evaluation evaluation = performEvaluation(testData, sparkNetwork);

		saveResult(evaluation, sparkNetwork.getNetwork());			
		
		doCleaning(sparkContext, sparkNetwork.getTrainingMaster());
	}

	private static void saveResult(Evaluation evaluation, MultiLayerNetwork multiLayerNetwork) throws IOException {
        if(saveDir != null && !saveDir.isEmpty()){
            File sd = new File(saveDir);
            if(!sd.exists()) {
                sd.mkdirs();
            }

            log.info("Saving network and evaluation stats to directory: {}", saveDir);
            saveModel(multiLayerNetwork);
            FileUtils.writeStringToFile(new File(saveDir, "evaulation.txt"), evaluation.stats(), StandardCharsets.UTF_8);
        }
        
      saveModel(multiLayerNetwork);
	}

	private static void saveModel(MultiLayerNetwork multiLayerNetwork) throws IOException {
		    File locationToSave = new File(saveDir);      
		    multiLayerNetwork.save(locationToSave, isSaveUpdater);
	}
	
    

	private static SparkDl4jMultiLayer loadResult(JavaSparkContext sparkContext, TrainingMaster<?, ?> trainingConfig) throws IOException {
		MultiLayerNetwork multiLayerNetwork = null;
        if(saveDir != null && !saveDir.isEmpty()){
        	File sd = new File(saveDir);
            if(!sd.exists()) {
                sd.mkdirs();
            }

            log.info("Loading network from directory: {}", saveDir);
            loadModel();
        } else {
        	log.info("Can't load network from directory: {}", saveDir);
        	log.info("Create default network");
        	multiLayerNetwork = new MultiLayerNetwork(NeuralNetworkConfig.getCnnNetwork(seed));
        }
        
        return new SparkDl4jMultiLayer(sparkContext, multiLayerNetwork, trainingConfig);
	}

	private static MultiLayerNetwork loadModel() throws IOException {
		File locationToSave = new File(saveDir); 
		return MultiLayerNetwork.load(locationToSave, isSaveUpdater);
	}

	private static void doCleaning(JavaSparkContext sparkContext, TrainingMaster<?, ?> trainingConfig) {
		trainingConfig.deleteTempFiles(sparkContext);

		log.info("***** Delete the temp training files *****");
	}

	private static Evaluation performEvaluation(JavaRDD<DataSet> testData, SparkDl4jMultiLayer sparkNetwork) {
		// Perform evaluation (distributed)
//	        Evaluation evaluation = sparkNet.evaluate(testData);
		Evaluation evaluation = sparkNetwork.doEvaluation(testData, 64, new Evaluation(10))[0]; // Work-around for 0.9.1
																							// bug: see
																							// https://deeplearning4j.org/releasenotes
		log.info("***** Evaluation *****");
		log.info(evaluation.stats());
		
		return evaluation;
	}

	private static void executeTraining(JavaRDD<DataSet> trainData, SparkDl4jMultiLayer sparkNet) {
		for (int i = 0; i < numEpochs; i++) {
			sparkNet.fit(trainData);
			log.info("Completed Epoch {}", i);
		}
	}

	private static TrainingMaster<?, ?> configureTraining() {
		// Configuration for Spark training: see http://deeplearning4j.org/spark for
		// explanation of these configuration options
		return new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker) // Each DataSet object:
																								// contains (by default)
				.averagingFrequency(averagingFrequency)
				.workerPrefetchNumBatches(workerPrefetchNumBatches) // Async prefetching: 2 examples per worker
				.batchSizePerWorker(batchSizePerWorker)
				.build();
	}



	public static void createNetwork(JavaSparkContext sc, String saveDirectory) throws IOException {
		saveDir = saveDirectory;
		boolean isSaved = false;
		createNetwork(sc, isSaved);
		
	}
}
