package pl.mariuszczarny.deepLearning.mnist;

import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

public class NeuralNetworkConfig {
	public static MultiLayerConfiguration getDeafultMultiLayerNetwork() {
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
						.nOut(15)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.activation(Activation.SOFTMAX)
						.nOut(10)
						.build())
				.build();
	}
	
	public static MultiLayerConfiguration getCnnNetwork() {
		int height = 28;    // height of the picture in px
        int width = 28;     // width of the picture in px
        int channels = 1;   // single channel for grayscale images
        int outputNum = 10; // 10 digits classification
        int batchSize = 54; // number of samples that will be propagated through the network in each iteration
        int nEpochs = 1;    // number of training epochs

        int seed = 1234;    // number used to initialize a pseudorandom number generator.
        
        // reduce the learning rate as the number of training epochs increases
        // iteration #, learning rate
        Map<Integer, Double> learningRateSchedule = new HashMap<>();
        learningRateSchedule.put(0, 0.06);
        learningRateSchedule.put(200, 0.05);
        learningRateSchedule.put(600, 0.028);
        learningRateSchedule.put(800, 0.0060);
        learningRateSchedule.put(1000, 0.001);
		
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005) // ridge regression value
                .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                    .nIn(channels)
                    .stride(1, 1)
                    .nOut(20)
                    .activation(Activation.IDENTITY)
                    .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                    .stride(1, 1) // nIn need not specified in later layers
                    .nOut(50)
                    .activation(Activation.IDENTITY)
                    .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                    .nOut(500)
                    .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .build())
                .setInputType(InputType.convolutionalFlat(height, width, channels)) // InputType.convolutional for normal image
                .build();
	}
}
