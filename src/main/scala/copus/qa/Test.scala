package copus.qa

import java.io.{File, PrintWriter}

import org.apache.spark.mllib.feature.HashingTF


object Test {
  def main(args: Array[String]): Unit = {
    val hashingTF = new HashingTF()
    println("\n\n=====>" + hashingTF.indexOf("Mandela") + "\n\n\n")
  }
}
