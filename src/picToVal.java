import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

public class picToVal {
	
	String filePath;
	double[][] normImg = null;
	int width = 0;
	int height = 0;
	String label;
	
	public picToVal(String imagePath) {
		filePath = imagePath;
		setValues();
		label = Character.toString(imagePath.charAt(0));
		//System.out.println("Learning: "+label);
	}


	private void setValues() {
		try {
			File file = new File("Training_Set/"+filePath);
		    BufferedImage img = ImageIO.read(file);
		    width = img.getWidth();
		    height = img.getHeight();
		    normImg = new double[width][height];
		    double total =0;
		    Raster raster = img.getData();
		    for (int i = 0; i < width; i++) {
		        for (int j = 0; j < height; j++) {
		        	double rawValue = raster.getSample(i, j, 0);
		        	normImg[i][j] = (255-rawValue);
		        	total += normImg[i][j];
		        }
		    }
		    double mean = total/784;
		    for (int i = 0; i < width; i++) {
		        for (int j = 0; j < height; j++) {
		        	normImg[i][j] = (normImg[i][j]-mean)/255;
		        }
		    }
		    //System.out.println("normalized");
		    //printVal(width, height, normImg);
	        
		}
		catch (IOException ex) {
		    ex.printStackTrace();
		}
	}
	
	
	public double[][] getImgVal(){
		return normImg;
	}
	
	private void printVal(int w, int h, double [][] values) {
		for (int j = 0; j < h; j++) {
        	System.out.println("");
        	for (int i = 0; i < w; i++) {
	            System.out.print(values[i][j]+" ");
	        }
	    } 
	}
	

}
