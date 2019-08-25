package pl.mariuszczarny.deepLearning.mnist;

import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
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
	
	public static MultiLayerConfiguration getCnnNetwork(int seed) {
		int height = 28;    // height of the picture in px
        int width = 28;     // width of the picture in px
        int channels = 1;   // single channel for grayscale images
        int outputNum = 10; // 10 digits classification
        
        Map<Integer, Double> learningRateScheduleMap = new HashMap<>();
        learningRateScheduleMap.put(0, 0.06);
        learningRateScheduleMap.put(200, 0.05);
        learningRateScheduleMap.put(600, 0.028);
        learningRateScheduleMap.put(800, 0.006);
        learningRateScheduleMap.put(1000, 0.001);
		
        MapSchedule learningRateSchedule = new MapSchedule(ScheduleType.ITERATION, learningRateScheduleMap);
		
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.00005)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(learningRateSchedule))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                    .nIn(channels)
                    .stride(1, 1)
                    .nOut(20)
                    .activation(Activation.IDENTITY)
                    .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(3, 3)
                    .stride(2, 2)
                    .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                    .stride(1, 1)
                    .nOut(50)
                    .activation(Activation.IDENTITY)
                    .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(3, 3)
                    .stride(2, 2)
                    .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                    .nOut(500)
                    .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .build())
                .setInputType(InputType.convolutionalFlat(height, width, channels))
                .build();
	}
}
