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

package pl.mariuszczarny.deepLearning;

import com.beust.jcommander.Parameter;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.evaluation.classification.Evaluation.Metric;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import pl.mariuszczarny.deepLearning.JCommanderUtils;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

/**
 * This is a local (single-machine) version of the Tiny ImageNet image classifier from TrainSpark.
 * See that example for details.
 *
 * Note that unlike the Spark training version, this local (single machine) version does not require the preprocessing
 * scripts to be run.
 *
 * @author Alex Black
 */
public class SparkLocal {
    public static Logger log = LoggerFactory.getLogger(SparkLocal.class);
    private ComputationGraph neuronNetwork;

    @Parameter(names = {"--numEpochs"}, description = "Number of epochs for training")
    private int numEpochs = 10;

    @Parameter(names = {"--saveDir"}, description = "If set, the directory to save the trained network")
    private String saveDir;

    public static void main(String[] args) throws IOException  {
        new SparkLocal().entryPoint(args);
    }

    public void entryPoint(String[] args) throws IOException {
        JCommanderUtils.parseArgs(this, args);

        int batchSize = 32;
        
        DataSetIterator iter = createDataPipeline(batchSize);
        neuronNetwork = createNetwork();

        //Reduce auto GC frequency for better performance
        int GCfrequency = 10000;
		Nd4j.getMemoryManager().setAutoGcWindow(GCfrequency);

        //Fit the network
        neuronNetwork.fit(iter, numEpochs);
        log.info("Training complete. Starting evaluation.");

        evaluateNetworkOnTestSet(batchSize);

        log.info("----- Examples Complete -----");
    }

	private void evaluateNetworkOnTestSet(int batchSize) throws IOException {
		DataSetIterator test = new TinyImageNetDataSetIterator(batchSize, DataSetType.TEST);
        test.setPreProcessor(new ImagePreProcessingScaler());   //Scale 0-255 valued pixels to 0-1 range
        Evaluation evaluation = new Evaluation(TinyImageNetDataSetIterator.getLabels(false), 5); //Set up for top 5 accuracy
        neuronNetwork.doEvaluation(test, evaluation);

        log.info("Evaluation complete");
        log.info(evaluation.stats());

        if(saveDir != null && !saveDir.isEmpty()){
            File file = new File(saveDir);
            if(!file.exists())
                file.mkdirs();

            log.info("Saving network and evaluation stats to directory: {}", saveDir);
            neuronNetwork.save(new File(saveDir, "trainedNet.bin"));
            FileUtils.writeStringToFile(new File(saveDir, "evaulation.txt"), evaluation.stats(), StandardCharsets.UTF_8);
        }
	}

	private ComputationGraph createNetwork() {
		ComputationGraph network = SparkTest.getNetwork();
		final int frequency = 50;
		final boolean reportScore = true;
        network.setListeners(new PerformanceListener(frequency, reportScore));
		return network;
	}

	private DataSetIterator createDataPipeline(int batchSize) {
		DataSetIterator iterator = new TinyImageNetDataSetIterator(batchSize);
        iterator.setPreProcessor(new ImagePreProcessingScaler());   //Scale 0-255 valued pixels to 0-1 range
		return iterator;
	}
}