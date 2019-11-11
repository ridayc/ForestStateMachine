package flib.io;

import java.io.File;
import java.io.Scanner;
java.util.regex.MatchResult;

public class TextReader {
	public static double[][] readTable(String filename, String delim){
		File file = new File(filename);
		Scanner s = new Scanner(file);
		s.findInLine(delim)