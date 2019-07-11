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

package pl.mariuszczarny.deepLearning.start;

import static pl.mariuszczarny.deepLearning.start.LocalSource.*;

import org.apache.spark.api.java.JavaSparkContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.beust.jcommander.Parameter;

import pl.mariuszczarny.deepLearning.mnist.MnistNetwork;
import pl.mariuszczarny.deepLearning.utils.JCommanderUtils;


/**
 * This is a local (single-machine) version of the Tiny ImageNet image classifier from TrainSpark.
 * See that example for details.
 *
 * Note that unlike the Spark training version, this local (single machine) version does not require the preprocessing
 * scripts to be run.
 *
 * @author Alex Black
 */
public class ApplicationRunner {
    public static Logger log = LoggerFactory.getLogger(ApplicationRunner.class);

    @Parameter(names = {"--numEpochs"}, description = "Number of epochs for training")
    private int numEpochs = 10;

    @Parameter(names = {"--saveDir"}, description = "If set, the directory to save the trained network")
    private String saveDir;

    public static void main(String[] args) throws Exception {
        new ApplicationRunner().entryPoint(args);
    }

    public void entryPoint(String[] args) throws Exception {
        JCommanderUtils.parseArgs(this, args);
        JavaSparkContext sc = JavaSparkHelper.initSparkContext();
        LocalSource localSource = MNIST;
        
        switch (localSource) {
		case MNIST:
			MnistNetwork.createNetwork(sc);
			break;
		case TINYIMAGENET:
			break;
		default:
			break;
        }
    }
}
