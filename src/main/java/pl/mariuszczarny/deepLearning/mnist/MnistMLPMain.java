package pl.mariuszczarny.deepLearning.mnist;

import org.apache.spark.api.java.JavaSparkContext;
import org.nd4j.linalg.factory.Nd4j;

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

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;

import pl.mariuszczarny.deepLearning.start.JavaSparkHelper;

/**
 * Train a simple/small MLP on MNIST data using Spark, then evaluate it on the
 * test set in a distributed manner
 *
 * Note that the network being trained here is too small to make proper use of
 * Spark - but it shows the configuration and evaluation used for Spark
 * training.
 *
 *
 * To run the example locally: Run the example as-is. The example is set up to
 * use Spark local by default. NOTE: Spark local should only be used for
 * development/testing. For data parallel training on a single machine (for
 * example, multi-GPU systems) instead use ParallelWrapper (which is faster than
 * using Spark for training on a single machine). See for example
 * MultiGpuLenetMnistExample in dl4j-cuda-specific-examples
 *
 * To run the example using Spark submit (for example on a cluster): pass
 * "-useSparkLocal false" as the application argument, OR first modify the
 * example by setting the field "useSparkLocal = false"
 *
 * @author Alex Black
 */
public class MnistMLPMain {
	@Parameter(names = "-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
	private boolean useSparkLocal = true;

	public static void main(String[] args) throws Exception {
		new MnistMLPMain().entryPoint(args);
	}

	protected void entryPoint(String[] args) throws Exception {
		useJCommand(args);
		Nd4j.getMemoryManager().setAutoGcWindow(5000);
		JavaSparkContext sc = JavaSparkHelper.initSparkContext(useSparkLocal);
		MnistNetwork.createNetwork(sc, false);
		//MnistNetwork.restoreNetwork(sc);

	}

	private void useJCommand(String[] args) {
		JCommander jcmdr = new JCommander(this);
		try {
			jcmdr.parse(args);
		} catch (ParameterException e) {
			jcmdr.usage();
			try {
				Thread.sleep(500);
			} catch (Exception e2) {
			}
			throw e;
		}
	}
}