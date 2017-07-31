package copus.corenlp;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class CoreNlpWrapper {
  
  static Random random = new Random();
  
  private static StanfordCoreNLP getPipeline() {
    return CoreNlpPipeline.getPipeline();
  }
  
  public static Document prepareText(String text) {
    return new Document(getPipeline().process(text));
  }//prepareText()
  
  public static double scoreDocument(Document document, Question question) {
    double score = 0.0;
    int[] indices = document.getTfIdfVector().indices();
    double[] values = document.getTfIdfVector().values();
    
    for(int hash : question.getTargetTermHashes()) {
      int index = Arrays.binarySearch(indices, hash);
      if(index >= 0)
        score += values[index];
    }
    return score;
  }//scoreDocument()
  
  public static double scoreSentence(Sentence sentence, Question question) {
    double score = 0.0;
    for(Token qtok : question.getTargetTerms()) {
      for(Token stok : sentence.getTokens()) {
        String qword = qtok.getLemma(), sword = stok.getLemma();
        if(qword.length() >= 3 && sword.length() >= 3 && qword.equals(sword))
          score += 1.0;
      }
    }
    return score;
  }//scoreSentence()
  
  public static String extractAnswer(Sentence sentence, Question question) {
    ArrayList<NamedEntityClass> objectives = new ArrayList<>();
    switch (question.getInterrogativeType()) {
      case WHO   :
      case WHAT  :
        objectives.add(NamedEntityClass.PERSON);
        objectives.add(NamedEntityClass.ORGANIZATION);
        break;
      case WHEN  :
        objectives.add(NamedEntityClass.DATE);
        objectives.add(NamedEntityClass.TIME);
        break;
      case WHERE :
        objectives.add(NamedEntityClass.LOCATION);
        break;
      default: return null;
    }
    NamedEntityClass[] targets = new NamedEntityClass[objectives.size()];
    objectives.toArray(targets);
    Token[] tokens = sentence.getTokens();
    int start = -1, end = -1;
    for (int i = 0; i < tokens.length; i++) {
      for (int j = 0; j < targets.length; j++) {
        if(tokens[i].getNec() == targets[j]) {
          if (start < 0) {
            start = i;
          }
          end = i;
        }
      }//inner for
    }//outer for
    if(start == -1)
      return null;
    StringBuffer sb = new StringBuffer();
    for(int i = start; i < end; i++)
      sb.append(tokens[i].getText() + " ");
    return sb.toString();
  }//extractAnswer()
  
  // main() used for unit testing only
  public static void main(String[] args) {

  }//main()
    
}//class CoreNlpWrapper
