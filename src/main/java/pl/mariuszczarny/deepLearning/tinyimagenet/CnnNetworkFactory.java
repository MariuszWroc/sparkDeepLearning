package pl.mariuszczarny.deepLearning.tinyimagenet;

import java.io.BufferedOutputStream;
import java.io.IOException;

import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.loader.impl.RecordReaderFileBatchLoader;
import org.deeplearning4j.datasets.fetchers.TinyImageNetFetcher;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold.AdaptiveThresholdAlgorithm;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.helper.DarknetHelper;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.AMSGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.enums.MeshBuildMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.beust.jcommander.Parameter;

public class CnnNetworkFactory {
	public static Logger log = LoggerFactory.getLogger(CnnNetworkFactory.class);
	
    /* --- Required Arguments -- */

    @Parameter(names = {"--dataPath"}, description = "Path (on HDFS or similar) of data preprocessed by preprocessing script." +
        " See PreprocessLocal or PreprocessSpark", required = true)
    private static String dataPath = "C:\\Users\\Mariusz\\Development\\workspace\\spark\\deepLearning";

    @Parameter(names = {"--masterIP"}, description = "Controller/master IP address - required. For example, 10.0.2.4", required = true)
    private static String masterIP = "0.0.0.0";

    @Parameter(names = {"--networkMask"}, description = "Network mask for Spark communication. For example, 10.0.0.0/16", required = true)
    private static String networkMask = "10.0.0.0/16";

    @Parameter(names = {"--numNodes"}, description = "Number of Spark nodes (machines)", required = true)
    private int numNodes = 2;
    
    /* --- Optional Arguments -- */
	
    @Parameter(names = {"--numEpochs"}, description = "Number of epochs for training")
    private static int numEpochs = 10;

    @Parameter(names = {"--minibatch"}, description = "Minibatch size (of preprocessed minibatches). Also number of" +
        "minibatches per worker when fitting")
    private static int minibatch = 32;

    @Parameter(names = {"--numWorkersPerNode"}, description = "Number of workers per Spark node. Usually use 1 per GPU, or 1 for CPU-only workers")
    private static int numWorkersPerNode = 1;

    @Parameter(names = {"--gradientThreshold"}, description = "Gradient threshold. See ")
    private static double gradientThreshold = 1E-3;

    @Parameter(names = {"--port"}, description = "Port number for Spark nodes. This can be any free port (port must be free on all nodes)")
    private static int port = 40123;
    
    @Parameter(names = {"--saveDirectory"}, description = "If set: save the trained network plus evaluation to this directory." +
            " Otherwise, the trained net will not be saved")
    private static String saveDirectory = "C:\\Users\\Mariusz\\Development\\tinyimagenet";
	
	public static void runNetwork(JavaSparkContext sc) throws IOException {
		//Set up TrainingMaster for gradient sharing training
        VoidConfiguration voidConfiguration = VoidConfiguration.builder()
            .unicastPort(port)                          // Should be open for IN/OUT communications on all Spark nodes
            .networkMask(networkMask)                   // Local network mask - for example, 10.0.0.0/16 - see https://deeplearning4j.org/docs/latest/deeplearning4j-scaleout-parameter-server
            .controllerAddress(masterIP)                // IP address of the master/driver node
            .meshBuildMode(MeshBuildMode.PLAIN)
            .build();
        TrainingMaster<?,?> tm = new SharedTrainingMaster.Builder(voidConfiguration, minibatch)
            .rngSeed(12345)
            .collectTrainingStats(false)
            .batchSizePerWorker(minibatch)              // Minibatch size for each worker
            .thresholdAlgorithm(new AdaptiveThresholdAlgorithm(gradientThreshold))     //Threshold algorithm determines the encoding threshold to be use. See docs for details
            .workersPerNode(numWorkersPerNode)          // Workers per node
            .build();


        ComputationGraph net = CnnNetworkFactory.getCCNNetwork();
        SparkComputationGraph sparkNet = new SparkComputationGraph(sc, net, tm);
        sparkNet.setListeners(new PerformanceListener(10, true));

        //Create data loader
        int imageHeightWidth = 64;      //64x64 pixel input
        int imageChannels = 3;          //RGB
        PathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader rr = new ImageRecordReader(imageHeightWidth, imageHeightWidth, imageChannels, labelMaker);
        rr.setLabels(new TinyImageNetDataSetIterator(1).getLabels());
        int numClasses = TinyImageNetFetcher.NUM_LABELS;
        RecordReaderFileBatchLoader loader = new RecordReaderFileBatchLoader(rr, minibatch, 1, numClasses);
        loader.setPreProcessor(new ImagePreProcessingScaler());   //Scale 0-255 valued pixels to 0-1 range


        //Fit the network
        String trainPath = dataPath + (dataPath.endsWith("/") ? "" : "/") + "train";
        JavaRDD<String> pathsTrain = SparkUtils.listPaths(sc, trainPath);
        for (int i = 0; i < numEpochs; i++) {
            log.info("--- Starting Training: Epoch {} of {} ---", (i + 1), numEpochs);
            sparkNet.fitPaths(pathsTrain, loader);
        }

        //Perform evaluation
        String testPath = dataPath + (dataPath.endsWith("/") ? "" : "/") + "test";
        JavaRDD<String> pathsTest = SparkUtils.listPaths(sc, testPath);
        Evaluation evaluation = new Evaluation(TinyImageNetDataSetIterator.getLabels(false), 5); //Set up for top 5 accuracy
        evaluation = (Evaluation) sparkNet.doEvaluation(pathsTest, loader, evaluation)[0];
        log.info("Evaluation statistics: {}", evaluation.stats());

        if (saveDirectory != null && saveDirectory.isEmpty()) {
            log.info("Saving the network and evaluation to directory: {}", saveDirectory);

            // Save network
            String networkPath = FilenameUtils.concat(saveDirectory, "network.bin");
            FileSystem fileSystem = FileSystem.get(sc.hadoopConfiguration());
            try (BufferedOutputStream os = new BufferedOutputStream(fileSystem.create(new Path(networkPath)))) {
                ModelSerializer.writeModel(sparkNet.getNetwork(), os, true);
            }

            // Save evaluation
            String evalPath = FilenameUtils.concat(saveDirectory, "evaluation.txt");
            SparkUtils.writeStringToFile(evalPath, evaluation.stats(), sc);
        }


        log.info("----- Example Complete -----");
	}
	
    public static ComputationGraph getCCNNetwork() {
        //This network: created for the purposes of this example. It is a simple CNN loosely inspired by the DarkNet
        // architecture, which was in turn inspired by the VGG16/19 networks
        //The performance of this network can likely be improved

        ISchedule lrSchedule = new MapSchedule.Builder(ScheduleType.EPOCH)
            .add(0, 8e-3)
            .add(1, 6e-3)
            .add(3, 3e-3)
            .add(5, 1e-3)
            .add(7, 5e-4)
            .build();

        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
            .convolutionMode(ConvolutionMode.Same)
            .l2(1e-4)
            .updater(new AMSGrad(lrSchedule))
            .weightInit(WeightInit.RELU)
            .graphBuilder()
            .addInputs("input")
            .setOutputs("output");

        DarknetHelper.addLayers(builder, 0, 3, 3, 32, 0);     //64x64 out
        DarknetHelper.addLayers(builder, 1, 3, 32, 64, 2);    //32x32 out
        DarknetHelper.addLayers(builder, 2, 2, 64, 128, 0);   //32x32 out
        DarknetHelper.addLayers(builder, 3, 2, 128, 256, 2);   //16x16 out
        DarknetHelper.addLayers(builder, 4, 2, 256, 256, 0);   //16x16 out
        DarknetHelper.addLayers(builder, 5, 2, 256, 512, 2);   //8x8 out

        builder.addLayer("convolution2d_6", new ConvolutionLayer.Builder(1, 1)
            .nIn(512)
            .nOut(TinyImageNetFetcher.NUM_LABELS)
            .weightInit(WeightInit.XAVIER)
            .stride(1, 1)
            .activation(Activation.IDENTITY)
            .build(), "maxpooling2d_5")
            .addLayer("globalpooling", new GlobalPoolingLayer.Builder(PoolingType.AVG).build(), "convolution2d_6")
            .addLayer("loss", new LossLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).build(), "globalpooling")
            .setOutputs("loss");

        ComputationGraphConfiguration config = builder.build();

        ComputationGraph network = new ComputationGraph(config);
        network.init();

        return network;
    }
}
