package flib.ij.celldetection;

import java.util.ArrayList;
import java.util.Set;
import java.util.Arrays;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import flib.math.VectorConv;
import flib.math.VectorFun;
import flib.math.VectorAccess;
import flib.math.UniqueSorted;
import flib.math.RankSort;
import flib.ij.stack.StackPixelArray;

public class CellSeparation {
	private ArrayList<int[]> cells = new ArrayList<int[]>();
	
	public CellSeparation(final ArrayList<ArrayList<Integer>> bloblist, final ImagePlus wsimp, double p){
		int w = wsimp.getWidth();
		int h = wsimp.getHeight();
		int s = w*h;
		int n = wsimp.getNSlices();
		int[] loc;
		double[] pixels, tpix, tpix2, merge;
		double[] wspixels = StackPixelArray.StackPixelArray(wsimp);
		int[] zlevel;
		int[] index,count, count2, count3, ind, ind2, cellindex, cs, cs2, cs3, csk, mergeind, tb, tbind, tix, tix2;
		int val,ncell;
		// list of how many over lapping pixels regions contain
		double[][] cr;
		// go through each blob individually
		for (int i=0; i<bloblist.size(); i++){
			// get the pixel locations of the blob
			loc = new int[bloblist.get(i).size()];
			for (int j=0; j<bloblist.get(i).size(); j++){
				loc[j] = bloblist.get(i).get(j);
			}
			// sort the pixels acording to their slice number
			RankSort RS = new RankSort(VectorConv.int2double(VectorFun.div(loc,s)),loc);
			zlevel = VectorConv.double2int(RS.getSorted());
			// sort the pixels locations accordingly
			loc = RS.getRank();
			// location of the pixels on a single slice
			index = VectorFun.mod(loc,s);
			// the watershed pixel values
			pixels = VectorAccess.access(wspixels,loc);
			// get the number of z levels
			count = UniqueSorted.unique(zlevel);
			// count the number of elements at each z level
			cs = new int[count.length];
			cs[0] = 0;
			for (int j=1; j<count.length; j++){
				cs[j] = count[j-1];
			}
			// start at the bottom level
			tpix = VectorAccess.access(pixels,0,count[0]);
			// sort the pixels according to their watershed group
			RS = new RankSort(tpix);
			tpix = RS.getSorted();
			ind = RS.getRank();
			tix = VectorAccess.access(index,ind);
			// find the indices of the starting and ending of each watershed region
			count2 = UniqueSorted.unique(tpix);
			// the watershed regions at the first level will definitely contain distinct cells
			ncell = count2.length;
			// prepare a list to store the relative cell number of the pixels in the blob
			cellindex = new int[loc.length];
			// start writing the relative indices of the cell in the correct location
			VectorAccess.write(cellindex,VectorAccess.access(ind,0,count2[0]),0);
			// size of the cell regions at the current z level
			cs2 = new int[ncell];
			cs2[0] = 0;
			Arrays.fill(tpix,0,count2[0],0);
			for (int j=1; j<ncell; j++){
				cs2[j] = count2[j-1];
				VectorAccess.write(cellindex,VectorAccess.access(ind,cs2[j],count2[j]),j);
				// we can use the fill function for tpix, since it's already been sorted
				Arrays.fill(tpix,cs2[j],count2[j],j);
			}
			// here comes the challenging code block
			// now we go through all the following z layers
			for (int j=1; j<count.length; j++){
				// we start similarly, as above
				tpix2 = VectorAccess.access(pixels,cs[j],count[j]);
				RS = new RankSort(tpix2);
				tpix2 = RS.getSorted();
				ind2 = RS.getRank();
				tix2 = VectorAccess.access(index,VectorFun.add(ind2,cs[j]));
				count3 = UniqueSorted.unique(tpix2);
				cs3 = new int[count3.length];
				cs3[0] = 0;
				Arrays.fill(tpix2,0,count3[0],0);
				for (int k=1; k<count3.length; k++){
					cs3[k] = count3[k-1];
					Arrays.fill(tpix2,cs3[k],count3[k],k);
				}
				// we merge the to index vectors together to check for pixels present in both layers
				merge = VectorAccess.vertCat2(VectorConv.int2double(tix),VectorConv.int2double(tix2));
				RS = new RankSort(merge);
				merge = RS.getSorted();
				mergeind = RS.getRank();
				tbind = UniqueSorted.unique(merge);
				csk= new int[tbind.length];
				csk[0] = 0;
				for (int k=1; k<tbind.length; k++){
					csk[k] = tbind[k-1];
				}
				// we prepare a matrix containing all possible layer overlaps for the regions in the two layers
				cr = new double[count2.length][count3.length];
				for (int k=0; k<tbind.length; k++){
					// if we observe two indices which are the same, we know we have overlapping pixel locations
					if ((tbind[k]-csk[k])>1){
						// we have to adjust the indices since we merged two vectors... but this helps us tell apart
						// which point belonged to which layer
						if (mergeind[csk[k]]>tpix.length){
							cr[(int)tpix[mergeind[csk[k]+1]]][(int)tpix2[mergeind[csk[k]]-tpix.length]]++;
						}
						else {
							cr[(int)tpix[mergeind[csk[k]]]][(int)tpix2[mergeind[csk[k]+1]-tpix.length]]++;
						}
					}
				}
				// calculate the minmum relative overlap between all regions
				for (int k=0; k<count2.length; k++){
					for (int l=0; l<count3.length; l++){
						if ((count2[k]-cs2[k])>(count3[l]-cs3[l])){
							cr[k][l]/=(double)(count2[k]-cs2[k]);
						}
						else  {
							cr[k][l]/=(double)(count3[l]-cs3[l]);
						}
					}
				}
				// go through all overlaps and put the regions with the largest overlap together
				boolean[] strike = new boolean[count3.length];
				for (int ii=0; ii<count3.length; ii++){
					double m = 0;
					int[] mind = new int[2];
					// find the maximum relative overlap
					for (int k=0; k<count2.length; k++){
						for (int l=0; l<count3.length; l++){
							if (cr[k][l]>m){
								m = cr[k][l];
								mind[0] = k;
								mind[1] = l;
							}
						}
					}
					// if the point with the maximum overlap exceeds the threshold value we associate the upper cluster to the lower one
					if (m>p){
						// give the upper cellindex the lower cellindex value
						VectorAccess.write(cellindex,VectorFun.add(VectorAccess.access(ind2,cs3[mind[1]],count3[mind[1]]),cs[j]),cellindex[ind[cs2[mind[0]]]+cs[j-1]]);
						// the regions can't overlap multiple times, so we all associated region values to zero
						for (int k=0; k<count2.length; k++){
							cr[k][mind[1]] = 0;
						}
						for (int k=0; k<count3.length; k++){
							cr[mind[0]][k] = 0;
						}
						// here we mark that the upper region was associated with a lower region
						strike[mind[1]] = true;
					}
				}
				// give all regions which didn't overlap (sufficiently) a new cell label
				for (int k=0; k<count3.length; k++){
					if (!strike[k]){
						VectorAccess.write(cellindex,VectorFun.add(VectorAccess.access(ind2,cs3[k],count3[k]),cs[j]),ncell);
						ncell++;
					}
				}
				// update all the lower layer values to the current layer
				count2 = count3;
				cs2 = cs3;
				tpix = tpix2;
				ind = ind2;
				tix = tix2;
			}
			// sort all cell indices according to number, as sort the location values accordingly
			RS = new RankSort(VectorConv.int2double(cellindex),loc);
			int[] indices = UniqueSorted.unique(RS.getSorted());
			loc = RS.getRank();
			// generate a new cell for each cell index
			this.cells.add(VectorAccess.access(loc,0,indices[0]));
			for (int j=1; j<indices.length; j++){
				this.cells.add(VectorAccess.access(loc,indices[j-1],indices[j]));
			}
		}

	}
	
	public ArrayList<int[]> getCells(){
		return this.cells;
	}
}