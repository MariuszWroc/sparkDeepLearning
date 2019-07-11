package pl.mariuszczarny.deepLearning.tinyimagenet;

import org.deeplearning4j.datasets.fetchers.TinyImageNetFetcher;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.model.helper.DarknetHelper;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AMSGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

public class NetworkFactory {
    public static ComputationGraph getCCNNetwork() {
        //This network: created for the purposes of this example. It is a simple CNN loosely inspired by the DarkNet
        // architecture, which was in turn inspired by the VGG16/19 networks
        //The performance of this network can likely be improved

        ISchedule lrSchedule = new MapSchedule.Builder(ScheduleType.EPOCH)
            .add(0, 8e-3)
            .add(1, 6e-3)
            .add(3, 3e-3)
            .add(5, 1e-3)
            .add(7, 5e-4).build();

        ComputationGraphConfiguration.GraphBuilder b = new NeuralNetConfiguration.Builder()
            .convolutionMode(ConvolutionMode.Same)
            .l2(1e-4)
            .updater(new AMSGrad(lrSchedule))
            .weightInit(WeightInit.RELU)
            .graphBuilder()
            .addInputs("input")
            .setOutputs("output");

        DarknetHelper.addLayers(b, 0, 3, 3, 32, 0);     //64x64 out
        DarknetHelper.addLayers(b, 1, 3, 32, 64, 2);    //32x32 out
        DarknetHelper.addLayers(b, 2, 2, 64, 128, 0);   //32x32 out
        DarknetHelper.addLayers(b, 3, 2, 128, 256, 2);   //16x16 out
        DarknetHelper.addLayers(b, 4, 2, 256, 256, 0);   //16x16 out
        DarknetHelper.addLayers(b, 5, 2, 256, 512, 2);   //8x8 out

        b.addLayer("convolution2d_6", new ConvolutionLayer.Builder(1, 1)
            .nIn(512)
            .nOut(TinyImageNetFetcher.NUM_LABELS)
            .weightInit(WeightInit.XAVIER)
            .stride(1, 1)
            .activation(Activation.IDENTITY)
            .build(), "maxpooling2d_5")
            .addLayer("globalpooling", new GlobalPoolingLayer.Builder(PoolingType.AVG).build(), "convolution2d_6")
            .addLayer("loss", new LossLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).build(), "globalpooling")
            .setOutputs("loss");

        ComputationGraphConfiguration conf = b.build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        return net;
    }
}
