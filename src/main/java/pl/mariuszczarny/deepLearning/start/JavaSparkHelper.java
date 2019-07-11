package pl.mariuszczarny.deepLearning.start;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

public class JavaSparkHelper {

	public static JavaSparkContext initSparkContext(boolean useSparkLocal) {
		SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("DL4J Spark MLP Example");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
		return sc;
	}
	
	public static JavaSparkContext initSparkContext() {
		SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        
        sparkConf.setAppName("DL4J Spark MLP Example");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
		return sc;
	}
}
