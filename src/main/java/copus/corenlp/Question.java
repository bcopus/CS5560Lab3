package copus.corenlp;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.SparseVector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Question implements Serializable {
  private String text;
  private Annotation annotation;
  private Token[] tokens;
  
  private InterrogativeType interrogativeType;
  private Token[] targetTerms;
  private int[] targetTermHashes;
  
  public Question(String text, HashingTF hasher) {
    this.text = text;
    annotation = CoreNlpPipeline.getPipeline().process(text);
    tokens = extractTokens();
    interrogativeType = extractInterrogativeType();
    targetTerms = extractTargetTerms();
    targetTermHashes = extractTargetTermHashes(hasher);
  }//Question()
  
  private Token[] extractTokens() {
    List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);
    Token[] tokensArray = new Token[tokens.size()];
    int i = 0;
    for(CoreLabel tok : tokens)
      tokensArray[i++] = new Token(tok);
    return tokensArray;
  }//extractTokens()
  
  private InterrogativeType extractInterrogativeType() {
    //  given a token with pos == "WP" or pos == "WRB", extract its lemma, and
    //  use it to determine the question type.
    for(Token tok : tokens) {
      if (tok.getPos() == PartOfSpeech.WP || tok.getPos() == PartOfSpeech.WRB) {
        switch (tok.getLemma().toLowerCase()) {
          case "who"   : return InterrogativeType.WHO;
          case "what"  : return InterrogativeType.WHAT;
          case "when"  : return InterrogativeType.WHEN;
          case "where" : return InterrogativeType.WHERE;
        }//switch
      }
    }
    return InterrogativeType.UNKNOWN;
  }//extractInterrogativeType()
  
  private Token[] extractTargetTerms() {
    //collect the nouns and verbs from the question
    ArrayList<Token> targets = new ArrayList<>();
    for(Token tok : tokens)
      if(tok.isNoun() || tok.isVerb())
        targets.add(tok);
    return targets.toArray(new Token[targets.size()]);
  }//extractTargetTerms()
  
  private int[] extractTargetTermHashes(HashingTF hasher) {
    int[] hashes = new int[targetTerms.length];
    for(int i = 0; i < hashes.length; i++)
      hashes[i] = hasher.indexOf(targetTerms[i].getLemma());
    return hashes;
  }//extractTargetTermHashes()
  
  public String getText() {
    return text;
  }
  
  public Token[] getTokens() {
    return tokens;
  }
  
  public InterrogativeType getInterrogativeType() {
    return interrogativeType;
  }
  
  public Token[] getTargetTerms() {
    return targetTerms;
  }
  
  public int[] getTargetTermHashes() {
    return targetTermHashes;
  }
}//class Question
