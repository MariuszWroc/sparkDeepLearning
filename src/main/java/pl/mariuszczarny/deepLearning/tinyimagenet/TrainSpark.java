/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package pl.mariuszczarny.deepLearning.tinyimagenet;

import java.io.BufferedOutputStream;
import java.io.IOException;

import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
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

import pl.mariuszczarny.deepLearning.utils.JCommanderUtils;


/**
 * This example trains a convolutional neural network image classifier on the Tiny ImageNet dataset using Apache Spark
 *
 * The Tiny ImageNet dataset is an image dataset of size 64x64 images, with 200 classes, and 500 images per class,
 * for a total of 100,000 images.
 *
 * Before running this example, you should do ONE (either) of the following to prepare the data for training:
 * 1. Run PreprocessLocal, and copy the output files to remote storage for your cluster (HDFS, S3, Azure storage, etc), OR
 * 2. Run PreprocessSpark on the tiny imagenet source files
 *
 * The CNN classifier trained here is trained from scratch without any pretraining. It is a custom network architecture
 * with 1,077,160 parameters based loosely on the VGG/DarkNet architectures. Improved accuracy is likely possible with
 * a larger network, better selection of hyperparameters, and more epochs.
 *
 * For further details on DL4J's Spark implementation, see the "Distributed Deep Learning" pages at:
 * https://deeplearning4j.org/docs/latest/
 *
 * A local (single machine) version of this example is available in TrainLocal
 *
 *
 * @author Alex Black
 */
public class TrainSpark {
    public static final Logger log = LoggerFactory.getLogger(TrainSpark.class);

    @Parameter(names = {"--sparkAppName"}, description = "App name for spark. Optional - can set it to anything to identify your job")
    private String sparkAppName = "DL4JTinyImageNetExample";

    public static void main(String[] args) throws Exception {
        new TrainSpark().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        JCommanderUtils.parseArgs(this, args);

        SparkConf conf = new SparkConf();
        conf.setAppName(sparkAppName);
        log.info(conf.toDebugString());
        JavaSparkContext sc = new JavaSparkContext(conf);
        CnnNetworkFactory.runNetwork(sc);
    }

	


}
