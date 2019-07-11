package pl.mariuszczarny.deepLearning.mnist;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
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

	public static void createNetwork(JavaSparkContext sc) throws IOException {
		DataSetIterator iterTrain = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
		DataSetIterator iterTest = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
		List<DataSet> trainDataList = new ArrayList<>();
		List<DataSet> testDataList = new ArrayList<>();
		while (iterTrain.hasNext()) {
			trainDataList.add(iterTrain.next());
		}
		while (iterTest.hasNext()) {
			testDataList.add(iterTest.next());
		}

		JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);
		JavaRDD<DataSet> testData = sc.parallelize(testDataList);

		// ----------------------------------
		// Create network configuration and conduct network training
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).activation(Activation.LEAKYRELU)
				.weightInit(WeightInit.XAVIER).updater(new Nesterovs(0.1))// To configure:
																			// .updater(Nesterovs.builder().momentum(0.9).build())
				.l2(1e-4).list().layer(new DenseLayer.Builder().nIn(28 * 28).nOut(500).build())
				.layer(new DenseLayer.Builder().nOut(100).build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.activation(Activation.SOFTMAX).nOut(10).build())
				.build();

		// Configuration for Spark training: see http://deeplearning4j.org/spark for
		// explanation of these configuration options
		TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker) // Each DataSet object:
																								// contains (by default)
																								// 32 examples
				.averagingFrequency(5).workerPrefetchNumBatches(2) // Async prefetching: 2 examples per worker
				.batchSizePerWorker(batchSizePerWorker).build();

		// Create the Spark network
		SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);

		// Execute training:
		for (int i = 0; i < numEpochs; i++) {
			sparkNet.fit(trainData);
			log.info("Completed Epoch {}", i);
		}

		// Perform evaluation (distributed)
//	        Evaluation evaluation = sparkNet.evaluate(testData);
		Evaluation evaluation = sparkNet.doEvaluation(testData, 64, new Evaluation(10))[0]; // Work-around for 0.9.1
																							// bug: see
																							// https://deeplearning4j.org/releasenotes
		log.info("***** Evaluation *****");
		log.info(evaluation.stats());

		// Delete the temp training files, now that we are done with them
		tm.deleteTempFiles(sc);

		log.info("***** Example Complete *****");
	}
}
