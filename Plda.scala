package com.tang

import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.{SparseMatrix, Vectors}
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

    val totalTermCount = doc.map(_._2).flatMap(_.toIterator).map(term => (term, 1L)).reduceByKey(_ + _)
    totalTermCount.cache()
    //save to hdfs
    totalTermCount.saveAsTextFile(args(1) + "/TotalTermCount")

    val reducedTermCount = totalTermCount.filter(_._2 > minDocFreq)
    reducedTermCount.cache()
    //save to hdfs
    reducedTermCount.saveAsTextFile(args(1) + "/ReducedTermCount")

    val term = reducedTermCount.map(_._1)
    val index = term.map(word => hashTF.indexOf(word))
    //reduced term's index of term index in hashDocTerm
    val indexArray = index.collect()

    val docTerm = hashDocTerm.mapValues { tf =>
      var indexFreqArray = new ArrayBuffer[(Int, Double)]()
      for (i <- Range(0, indexArray.size)) {
        val tfFreq = tf.apply(indexArray(i))
        //sparse vector, filter tfFreq=0.0 elements, need tfFreq >= 1.0
        if (tfFreq > 0.5)
          indexFreqArray += (i -> tfFreq)
      }
      Vectors.sparse(indexArray.size, indexFreqArray)
    }

    docTerm.cache()
    //save to hdfs
    docTerm.saveAsTextFile(args(1) + "/DocTermMatrix")

    // Cluster the documents into three topics using LDA
    val distributedLdaModel: DistributedLDAModel = new LDA()
      .setK(topicNum)
      .setAlpha(alpha)
      .setBeta(beta)
      .setMaxIterations(maxIterations)
      .run(docTerm).asInstanceOf[DistributedLDAModel]

    val topicDistribution = distributedLdaModel.topicDistributions
    //save to hdfs
    topicDistribution.saveAsTextFile(args(1) + "/TopicDistribution")

    // Output the term-topics
    val termTopicMatrix = distributedLdaModel.topicsMatrix
    val topicColArray = termTopicMatrix.toString(termTopicMatrix.numRows, termTopicMatrix.numCols * 24).split("\n")
    val topicCol = sc.parallelize(topicColArray, 2)
    //save to hdfs
    topicCol.saveAsTextFile(args(1) + "/TermTopicMatrix")

    sc.stop()
  }
}
