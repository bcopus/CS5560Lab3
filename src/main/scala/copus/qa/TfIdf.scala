package copus.qa

import copus.corenlp.{CoreNlpWrapper, Document, Question}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.rdd.RDD

object TfIdf {

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\winutils")
    val sparkConf = new SparkConf().setAppName("QASystemLab2").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)

    val documents = sc.objectFile[Document]("documents.obj")

    //process each document into a sequence of lemmatized tokens
    val documentSequences = documents.map(doc => {
      doc.getTokens.map(tok => {
        tok.getLemma
      }).toSeq
    })
//    documentseq.foreach(doc => println(doc))

    val hashingTF = new HashingTF()

    val termFrequencies = hashingTF.transform(documentSequences)

    println(termFrequencies.getClass)
    termFrequencies.cache()
    termFrequencies.saveAsObjectFile("termFrequencyHashTable.obj")

//    println(hashingTF.indexOf("Mandela"))
    val idf = new IDF().fit(termFrequencies) //returns IDFModel
    val tfIdf = idf.transform(termFrequencies) //returns TF-IDF SparseVectors

    for (i : Int <- 0 until documents.count().toInt) {
      documents.collect.apply(i).setTfIdfVector(tfIdf.collect.apply(i).toSparse)
    }
    tfIdf.foreach(term => println(term))



//        val tfidfvalues = tfIdf.flatMap(f => {
//          val ff: Array[String] = f.toString.replace(",[", ";").split(";")
//          val values = ff(2).replace("]", "").replace(")", "").split(",")
//          values
//        })
//
//        val tfidfindex = tfIdf.flatMap(f => {
//          val ff: Array[String] = f.toString.replace(",[", ";").split(";")
//          val indices = ff(1).replace("]", "").replace(")", "").split(",")
//          indices
//        })
//
//        tfidfvalues.foreach(f => println(f))
//
//    val tfIdfTermHashes = tfIdf.flatMap(tfIdfVec => tfIdfVec.apply(1))
//
//        val tfidfData = tfidfindex.zip(tfidfvalues)
//
//        var hm = new HashMap[String, Double]
//
//        tfidfData.collect().foreach(f => {
//          hm += f._1 -> f._2.toDouble
//        })
//
//        val mapp = sc.broadcast(hm)
//
//        val documentData = documentseq.flatMap(_.toList)
//        val dd = documentData.map(f => {
//          val i = hashingTF.indexOf(f)
//          val h = mapp.value
//          (f, h(i.toString))
//        })
//
//        val dd1 = dd.distinct().sortBy(_._2, false)
//        dd1.take(20).foreach(f => {
//          println(f)
//        })

  }//main()
}
