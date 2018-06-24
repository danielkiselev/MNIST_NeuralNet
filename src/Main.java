import java.awt.Color;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import javax.imageio.ImageIO;

public class Main {

	
	final static File dir = new File("Training_Set/");
	static int [] hiddenLayerSize = {50};
	static String[] outputArr = {"0","1","2","3","4","5","6","7","8","9"};
//int inputSize, int hiddenLayerCount, int [] hiddenLayerSize, int outputSize, String[] outputArr, double learningRate
	static NeuralNet nn = new NeuralNet(784,1, hiddenLayerSize, 10, outputArr, .5 );
	static int totalSize = 0;
	
	public static void main(String[] args) {
		nn.Load();
		
		/*
        for (int j = 0; j < 40000; j++) { 
        	if(j%100 == 0) {
        		System.out.println("Learning Image#: "+ (j+1));
        	}
        	Train(getImage());
        	
        }
		*/
      
        for (int j = 0; j < 50; j++) {
        	Test(getImage());
        }
        
        double accuracy = nn.correct/50;
        System.out.println("Accuracy"+accuracy); 
        
        /*
        if(accuracy>=.7) {
        	System.out.println("Saving config");
        	nn.SaveNN();
        	System.out.println("Save Complete");
        }
        */
        
	}
	
	private static void Train(String name) {
		
		picToVal pic = new picToVal(name);
		nn.Learn(Flatten(pic),pic.label);
	
		//c2.convolve(pL1.actMap, pL1.pWidth, pL1.pHeight);
		//poolLayer pL2 = new poolLayer(c2.actMap, c2.width, c2.height);
		//nn.learnImgData(pL2.actMap, pL2.pWidth, pL2.pHeight, pic.label);
		
	}
	
	
	private static double[] Flatten (picToVal pic) {
		int picH = pic.height;
		int picW = pic.width;
		totalSize = picH*picW;
		double [] img1D = new double [totalSize];
		int counter = 0;
	    for (int i = 0; i < picH; i++) {
	    	for (int j = 0; j < picW; j++) {
	    		img1D[counter] = pic.normImg[j][i];
	    		counter++;
	    	}
	    }
		return img1D;
	}
	
	
	private static void Test(String name) {
		
		picToVal pic = new picToVal(name);
		nn.Guess(Flatten(pic), pic.label);

	}

	
	private static String getImage(){
		File[] files = dir.listFiles();

		Random rand = new Random();

		File file = files[rand.nextInt(files.length)];
		return(file.getName());
		
	}
	
}
