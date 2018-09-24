/* 
	--- IN PROGRESS ---
	This project reads text from two different authors and uses a naive bayes classifier to determine the author from a random piece of text written by one of the two authors. 
	
	ERROR: The naive bayes algorithm favors the author that has the least amount of unique words similiar to the random text. 
 */

import java.util.LinkedHashMap;
import java.util.Map;

public class AdvancedBayes {
	//HashMaps storing each sentences words and frequency of the word's occurrences
	public static LinkedHashMap<String, Double> personA = new LinkedHashMap<String, Double>();
	public static LinkedHashMap<String, Double> personB = new LinkedHashMap<String, Double>();
	
	//Training Text from Hemmingway
	public static String personAText = "THE FIRST FOUR STORIES ARE THE LAST ones I have written. The others follow in the order in which they were originally published. The first one I wrote was Up in Michigan, written in Paris in 1921. The last was Old Man at the Bridge, cabled from Barcelona in April of 1938.";
	
	//Training Text from Poe
	public static String personBText = "Once upon a midnight dreary, while I pondered, weak and weary, Over many a quaint and curious volume of forgotten lore While I nodded, nearly napping, suddenly there came a tapping, As of some one gently rapping, rapping at my chamber door. Tis some";

	//Test Text from Hemmingway
	/* 	public static String inputText = "There are many kinds of stories in this book. I hope that you will find some that you like. Reading them over, the ones I liked the best, outside of those that have achieved some notoriety so that school teachers include them in story collections that their pupils have to buy in story courses";
	*/	 
	
	//Test Text from Poe
	public static String inputText = "Ah, distinctly I remember it was in the bleak December; And each separate dying ember wrought its ghost upon the floor. Eagerly I wished the morrow; vainly I had sought to borrow From my books surcease of sorrow—sorrow for the lost Lenore For the rare";
	 
	//Formats texts
	public static String [] personAInput = personAText.toLowerCase().replaceAll("[^a-zA-Z0-9\\s]", "").split(" ");
	public static String [] personBInput = personBText.toLowerCase().replaceAll("[^a-zA-Z0-9\\s]", "").split(" ");
	public static String [] input = inputText.toLowerCase().replaceAll("[^a-zA-Z0-9\\s]", "").split(" ");
	
	
	public static void main(String [] args) {
		initializeFeatures();
		System.out.println(compareLabels(computeBayes(input)));
	}
	
	//Fills hashmaps with words and word frequencies
	public static void initializeFeatures() {
		//Person A
		for(String keys : personAInput) {
			if(personA.get(keys) != null) {
				personA.put(keys, personA.get(keys) + 1.00);
			} else {
				personA.put(keys, 1.000);
			}
		}
		for(Map.Entry<String, Double> entry : personA.entrySet()) {
			personA.put(entry.getKey(), entry.getValue()/personAInput.length);
		}
		System.out.println(personA);
			
		//Person B
		for(String keys : personBInput) {
			if(personB.get(keys) != null) {
				personB.put(keys, personB.get(keys) + 1.00);
			} else {
				personB.put(keys, 1.000);
			}
		}
		for(Map.Entry<String, Double> entry : personB.entrySet()) {
			personB.put(entry.getKey(), entry.getValue()/personBInput.length);
		}
		System.out.println(personB);
	}
	
	//Reads words from hashmaps and computes values using naive bayes
	public static LinkedHashMap<String, Double> computeBayes(String [] words) {
		double personASum = 0;
		double personBSum = 0;
		int index = 1;
		int counter = 0;
		for(String word : words) {
			if(personA.get(word)!= null) {
				counter++;
				System.out.println(word + ", " + counter);
				if(index > 0) personASum = personA.get(word);
				else personASum *= personA.get(word);
				index--;
			}
		}
		personASum *= 0.5;
		System.out.println("Person A has " + counter + " matching words");
		index = 1;
		counter = 0;
		for(String word : words) {
			if(personB.get(word)!= null) {
				counter++;
				System.out.println(word + ", " + counter);
				if(index > 0) personBSum = personB.get(word);
				else personBSum *= personB.get(word);
				index--;
			}
		}
		System.out.println("Person B has " + counter + " matching words");
		personBSum *= 0.5;
		
		//in case there is no match in words
		/* if(personASum > 0) personASum = 1/personASum;
		else personASum = 0.99999;
		if(personBSum > 0) personBSum = 1/personBSum;
		else personBSum = 0.99999; */		

		/* personASum = 1/personASum;
		personBSum = 1/personBSum; */
		
		double total = personASum + personBSum;
		personASum /= total;
		personBSum /= total;
	
		//Creates a new hashmap storing the labels for the two authors and their respective naive bayes scores.
		LinkedHashMap<String, Double> labels = new LinkedHashMap<String, Double>();
		labels.put("Person A", personASum);
		labels.put("Person B", personBSum);
	
		System.out.println("Person A : " + personASum*100 + "%" + ", " + "Person B : " + personBSum*100 + "%");
		return labels;
	}
	
	//Reads label hashmap and returns the person with the higher naive bayes score
	public static String compareLabels(LinkedHashMap<String, Double> labels) {
		if(labels.get("Person A") > labels.get("Person B")) return "Person A";
		else if(labels.get("Person A") < labels.get("Person B")) return
		"Person B";
		else return "Uncertain";
	}
	
	
}