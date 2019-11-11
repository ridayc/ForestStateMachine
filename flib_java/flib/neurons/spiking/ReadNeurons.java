package flib.neurons.spiking;

import java.io.ObjectInputStream;
import java.io.FileInputStream;
import java.io.ObjectOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.File;
import java.nio.channels.FileChannel;
import java.lang.ClassNotFoundException;
import flib.neurons.spiking.CombinedNetwork;

public class ReadNeurons {
	public static CombinedNetwork readNeurons(String filename){
		CombinedNetwork neuronset = null;
		try {
			FileInputStream in = new FileInputStream(filename);
			ObjectInputStream reader = new ObjectInputStream(in);
			neuronset = (CombinedNetwork) reader.readObject();
		}
		catch (IOException e){
			e.printStackTrace();
			return null;
		}
		catch (ClassNotFoundException c){
			c.printStackTrace();
			return null;
		}
		return neuronset;
	}
	
	public static void writeNeurons(String filename, final CombinedNetwork neuronset){
		try {
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
			out.writeObject(neuronset);
			out.close();
		} 
		catch(IOException e){}
	}
	
	public static void copyFileIO(String sourcePath, String destPath)
	throws IOException {
		File source = new File(sourcePath);
		// we can only copy if the source file exists
		if (source.isFile()){
			File dest = new File(destPath);
			// if the target file already exists we delete and overwrite it
			if (dest.isFile()){
				dest.delete();
			}
			dest.createNewFile();
			FileChannel inputChannel = null;
			FileChannel outputChannel = null;
			try {
				inputChannel = new FileInputStream(source).getChannel();
				outputChannel = new FileOutputStream(dest).getChannel();
				outputChannel.transferFrom(inputChannel, 0, inputChannel.size());
			} 
			finally {
				inputChannel.close();
				outputChannel.close();
			}
		}
	}
	
	public static void copyFile(String sourcePath, String destPath){
		try{
			copyFileIO(sourcePath,destPath);
		}
		catch(IOException e){}
	}
}
