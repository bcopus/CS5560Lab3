package copus.qa

import java.io.{File, PrintWriter}

import copus.corenlp.{CoreNlpWrapper, Document, Utilities}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.{HashingTF, IDF}

object TfIdfCorpusPrepper {

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\winutils")
    val sparkConf = new SparkConf().setAppName("QASystemLab3").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)

    val files = sc.wholeTextFiles("corpus")
    val documents = files.map(file => CoreNlpWrapper.prepareText(file._2)).toArray()
    val docsRDD = sc.parallelize(documents)

    //process each document into a sequence of lemmatized tokens
    val documentSequences = docsRDD.map(doc => {
      doc.getTokens.map(tok => {
        tok.getLemma
      }).toSeq
    })
    documentSequences.foreach(doc => println(doc))

    val hashingTF = new HashingTF()
    val termFrequencies = hashingTF.transform(documentSequences)
    termFrequencies.cache()
    termFrequencies.saveAsObjectFile("termFrequencyHashTable.obj")

    val idf = new IDF().fit(termFrequencies) //returns IDFModel
    val tfIdf = idf.transform(termFrequencies) //returns TF-IDF SparseVectors

    val dd = docsRDD.collect()
    val tt = tfIdf.collect()

    for (i : Int <- 0 until docsRDD.count().toInt) {
      dd.apply(i).setTfIdfVector(tt.apply(i).toSparse)
    }
    sc.parallelize(dd).saveAsObjectFile("documents.obj")

  }//main()

}//CorpusPrepper
