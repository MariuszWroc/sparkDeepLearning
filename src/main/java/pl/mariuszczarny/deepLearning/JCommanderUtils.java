package pl.mariuszczarny.deepLearning;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.ParameterException;

public class JCommanderUtils {
	public static final Logger log = LoggerFactory.getLogger(JCommanderUtils.class);

    private JCommanderUtils(){ }

    public static void parseArgs(Object obj, String[] args){
        JCommander jcmdr = new JCommander(obj);
        try {
            jcmdr.parse(args);
        } catch (ParameterException parameterException) {
            jcmdr.usage();  //User provides invalid input -> print the usage info
                try {
					Thread.sleep(500);
				} catch (InterruptedException interruptedException) {
					log.warn(interruptedException.getMessage(), interruptedException);
				}
            throw parameterException;
        }
    }
}
