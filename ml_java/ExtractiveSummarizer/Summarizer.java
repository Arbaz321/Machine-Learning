/* 
	(IN PROGRESS) Text Summarizer following Word Frequency Model: Reads a text and prints the sentence with the most frequenctly used words.

	TODO: 
		- Incorporate ability to read from text files rather than manually inputting text
		- Find a way to incorporate semanticity in algorithm
 */

//read in a text
//seperate the text into paragraphs
//separate paragraphs into sentences
//separate sentences into words with spaces (tokenize)

//go through every word in the text
//store the word and it's frequency in a dictionary

//go back through the text and assign values to each sentence as the sum of the word scores.
//store the sentence and the score in a dictionary

//go through the dictionary and determine the "k" sentences with the most scores
//print them in the order they come in (linkedhashmap).

import java.util.Map;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.LinkedList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Summarizer {	
	public static String Text;
	public static LinkedHashSet<String> stopwords = new LinkedHashSet<String>();
	public static String [] tokenizedWords;
	public static List <String> sentences = new LinkedList<String>();
	public static Map<String, Double> wordBank = new LinkedHashMap<String, Double>();
	public static Map<String, Double> sentenceBank = new LinkedHashMap<String, Double>();
	public static double totalTextScore = 0;
	
	public static void main(String [] args) {
		readText();
		readStopWords();
		tokenizedWords = tokenizeText(Text);
		buildWordBank(tokenizedWords);
		buildSentenceBank(Text);
		computeSentenceScores();
		summarizeText();
	}

	public static void readText() {
		// The name of the file to open.
        String fileName = "testText.txt";
        // This will reference one line at a time
        String line = null;
        try {
            // FileReader reads text files in the default encoding.
            FileReader fileReader = 
                new FileReader(fileName);

            // Always wrap FileReader in BufferedReader.
            BufferedReader bufferedReader = 
                new BufferedReader(fileReader);

            while((line = bufferedReader.readLine()) != null) {
                Text += line;
            }   

            // Always close files.
            bufferedReader.close();         
        }
        catch(FileNotFoundException ex) {
            System.out.println(
                "Unable to open file '" + 
                fileName + "'");                
        }
        catch(IOException ex) {
            System.out.println(
                "Error reading file '" 
                + fileName + "'");                  
            // Or we could just do this: 
            // ex.printStackTrace();
        }
	}
	
	public static void readStopWords() {
		// The name of the file to open.
        String fileName = "stopwords.txt";
        // This will reference one line at a time
        String line = null;
        try {
            // FileReader reads text files in the default encoding.
            FileReader fileReader = 
                new FileReader(fileName);

            // Always wrap FileReader in BufferedReader.
            BufferedReader bufferedReader = 
                new BufferedReader(fileReader);

            while((line = bufferedReader.readLine()) != null) {
                stopwords.add(line);
            }   

            // Always close files.
            bufferedReader.close();         
        }
        catch(FileNotFoundException ex) {
            System.out.println(
                "Unable to open file '" + 
                fileName + "'");                
        }
        catch(IOException ex) {
            System.out.println(
                "Error reading file '" 
                + fileName + "'");                  
            // Or we could just do this: 
            // ex.printStackTrace();
        }
	}
	
	//build an array of strings containing all individual words
	public static String [] tokenizeText(String text) {
		return text.toLowerCase().replaceAll("[^a-zA-Z0-9\\s]", "").split(" ");
	}
	
	//stores words from array into linkedhashmap with their frequencies
	public static void buildWordBank(String [] words) {
		for(String word : words) {
			if(!stopwords.contains(word)) {
				if(wordBank.get(word) != null) {
					wordBank.put(word, wordBank.get(word)+1.00);
				} else {
					wordBank.put(word, 1.00);
				}
			}
			totalTextScore++;
		}
	}
	
	//construct sentences: sentences should be in a linkedhashmap with their scores
	public static void buildSentenceBank(String text) {
		String sentenceBuild = "";
		text = text.toLowerCase();
		for(int i = 0; i < text.length(); i++) {
			if(text.charAt(i) == '.') {
				if(sentenceBuild.charAt(0) == ' ') {
					sentenceBuild = sentenceBuild.substring(1, sentenceBuild.length());
				}
				sentences.add(sentenceBuild);
				sentenceBuild = "";
			} else {
				sentenceBuild += text.charAt(i);
			}
		}
	}
	
	//compute sentence scores and save into sentenceBank
	public static void computeSentenceScores() {
		for(String sentence : sentences) {
			String [] tokenizedSentences = tokenizeText(sentence);
			double sentenceScore = 0;
			for(String word : tokenizedSentences) {
				if(wordBank.get(word)!= null) {
					sentenceScore += wordBank.get(word);
				}
			}
			sentenceBank.put(sentence, sentenceScore/totalTextScore);
		}
		
	}
	
	//print sentence with the highest sentence score
	public static void summarizeText() {
		int k = 3;
		
		double [] scores = new double [sentenceBank.size()];
		int index = 0;
		for(String score : sentenceBank.keySet()) {
			scores[index] = sentenceBank.get(score);
			index++;
		}
		
		Arrays.sort(scores);
		
		for(int i = 0; i < k; i++) {
			for(Map.Entry<String, Double> entry : sentenceBank.entrySet()) {
				if(entry.getValue() == scores[scores.length-1-i] && k <= sentenceBank.size()) {
					System.out.println("Sentence Score: " + entry.getValue());
					System.out.println(entry.getKey());
				}
			}
		}
	}
}




















