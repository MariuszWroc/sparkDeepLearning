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
import org.nd4j.evaluation.classification.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.beust.jcommander.Parameter;

public class MnistNetwork {
	private static final Logger log = LoggerFactory.getLogger(MnistNetwork.class);

	@Parameter(names = "-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
	private boolean useSparkLocal = true;

	@Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
	private static int batchSizePerWorker = 16;

	@Parameter(names = "-numEpochs", description = "Number of epochs for training")
	private static int numEpochs = 2;
	
	private static String saveDir;
	
	private static JavaRDD<DataSet> createRDD(JavaSparkContext sc) throws IOException {
		int seed = 12345;
		List<DataSet> dataList = new ArrayList<>();
		DataSetIterator dataSetIterator = new MnistDataSetIterator(batchSizePerWorker, true, seed);
		
		while (dataSetIterator.hasNext()) {
			dataList.add(dataSetIterator.next());
		}
		
		return sc.parallelize(dataList);
	}

	public static void createNetwork(JavaSparkContext sparkContext) throws IOException {
		JavaRDD<DataSet> trainingData = createRDD(sparkContext);
		JavaRDD<DataSet> testData = createRDD(sparkContext);
		MultiLayerConfiguration networkConfig = configureMultiLayerNetwork();
		TrainingMaster<?, ?> trainingConfig = configureTraining();
		SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sparkContext, networkConfig, trainingConfig);

		executeTraining(trainingData, sparkNetwork);
		Evaluation evaluation = performEvaluation(testData, sparkNetwork);

		saveResult(evaluation, sparkNetwork);
		doCleaning(sparkContext, trainingConfig);
	}

	private static void saveResult(Evaluation evaluation, SparkDl4jMultiLayer sparkNetwork) throws IOException {
        if(saveDir != null && !saveDir.isEmpty()){
            File sd = new File(saveDir);
            if(!sd.exists())
                sd.mkdirs();

            log.info("Saving network and evaluation stats to directory: {}", saveDir);
            sparkNetwork.getNetwork().save(new File(saveDir, "trainedNet.bin"));
            FileUtils.writeStringToFile(new File(saveDir, "evaulation.txt"), evaluation.stats(), StandardCharsets.UTF_8);
        }
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
																								// 32 examples
				.averagingFrequency(5)
				.workerPrefetchNumBatches(2) // Async prefetching: 2 examples per worker
				.batchSizePerWorker(batchSizePerWorker)
				.build();
	}

	private static MultiLayerConfiguration configureMultiLayerNetwork() {
		// ----------------------------------
		// Create network configuration and conduct network training
		return new NeuralNetConfiguration.Builder()
				.seed(12345)
				.activation(Activation.LEAKYRELU)
				.weightInit(WeightInit.XAVIER)
				.updater(new Nesterovs(0.1))// To configure:
																			// .updater(Nesterovs.builder().momentum(0.9).build())
				.l2(1e-4)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(28 * 28).nOut(500)
						.build())
				.layer(new DenseLayer.Builder()
						.nOut(100)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.activation(Activation.SOFTMAX)
						.nOut(10)
						.build())
				.build();
	}

	public static void createNetwork(JavaSparkContext sc, String saveDirectory) throws IOException {
		saveDir = saveDirectory;
		createNetwork(sc);
		
	}
}
