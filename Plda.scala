package com.tang

import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark._
import org.apache.spark.SparkContext
import scala.collection.mutable.ArrayBuffer

/**
 * Created by Tang Lizhe on 2015/9/11.
 */
object Plda {
  def main(args: Array[String]) {
    // Load and parse the data
    val topicNum = 20
    val alpha = -1
    val beta = -1
    val maxIterations = 150
    val minDocFreq = 10

    val conf = new SparkConf().setAppName("PLDA").set("spark.executor.memory", "2g")
    val sc = new SparkContext(conf)

    val file = sc.textFile(args(0))
    val doc = file.map { line =>
      val w = line.trim.split(";;;;;", 2)
      val docno = w(0).trim.toLong
      val terms = w(1).trim.split(" ")
      (docno, terms)
    }
    doc.cache()

    val hashTF = new HashingTF(1 << 30)
    val hashDocTerm = hashTF.transform(doc.mapValues(_.toIterable))

    val termCount = doc.map(_._2).flatMap(_.toIterator).map(term => (term, 1L)).reduceByKey(_ + _)
    termCount.cache()
    //save to hdfs
    termCount.saveAsTextFile(args(1) + "/termCount")

    val term = termCount.filter(_._2 > minDocFreq).map(_._1)
    //print all terms(has filter)
    /*    for (word <- termArray) {
          print(word + " ")
        }
        println()*/

    val index = term.map { word =>
      hashTF.indexOf(word)
    }
    val indexArray = index.collect() //index array of term's index in hashDocTerm

    //broadcast
    //val bcIndexArray = hashDocTerm.context.broadcast(indexArray)

    val docTerm = hashDocTerm.mapValues { tf =>
      //val thisIndexArray = bcIndexArray.value
      var indexFreqArray = new ArrayBuffer[(Int, Double)]()
      for (i <- Range(0, indexArray.size)) {
        val tfFreq = tf.apply(indexArray(i))

        if (tfFreq > 0.5) //sparse vector, need tfFreq > 0.0
          indexFreqArray += (i -> tfFreq)
      }
      Vectors.sparse(indexArray.size, indexFreqArray)
    }

    docTerm.cache()
    //save to hdfs
    docTerm.saveAsTextFile(args(1) + "/docTermMatrix")
    /*    val docTermMatrix = docTerm.map(_._2).collect()
        for (doc <- Range(0, docTermMatrix.size)) {
          val v = docTermMatrix(doc)
          for (term <- Range(0, v.size)) {
            print(v(term) + " ")
          }
          println()
        }*/

    // Cluster the documents into three topics using LDA
    val ldaModel: LDAModel = new LDA().setK(topicNum).setAlpha(alpha).setBeta(beta).setMaxIterations(maxIterations).run(docTerm)
    //save to hdfs

    // Output topics. Each is a distribution over terms (matching term count vectors)
    val termTopicMatrix = ldaModel.topicsMatrix
    //save to hdfs

    for (topic <- Range(0, topicNum)) {
      print("Topic " + topic + ":")
      for (term <- Range(0, ldaModel.vocabSize)) {
        print(" " + termTopicMatrix(term, topic))
      }
      println()
    }

    //  Save and load model.
    //    ldaModel.save(sc, "myLDAModel")
    //    val sameModel = new DistributedLDAModel().load(sc, "myLDAModel")
    //    new DistributedLDAModel().topicDistributions
    sc.stop()
  }
}
