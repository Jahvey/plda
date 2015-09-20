package com.tang

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.storage.StorageLevel
import scala.collection.mutable.ArrayBuffer

/**
 * Created by Tang Lizhe on 2015/9/11.
 */
object Plda {
  def main(args: Array[String]) {
    val topicNum = args(0).toInt
    val alpha = args(1).toDouble
    val beta = args(2).toDouble
    val maxIterations = args(3).toInt
    val minDocThreshold = args(4).toInt
    val hdfsHomeDir = "hdfs://node-25:9000/user/solo/"
    val inputPath = hdfsHomeDir + args(5)
    val outputPath = hdfsHomeDir + args(6)

    /**
     * If the word count value less than minDocFreq, the word will be filtered and will not be used
     * to calculate in following spark MLlib LDA computing.
     */
    val minDocFreq = minDocThreshold

    val conf = new SparkConf().setAppName("Spark LDA")
    val sc = new SparkContext(conf)

    /**
     * Let all documents from input file stored by doc:RDD[(Long, Array[String])].
     * Long position store docno.
     * Array[String] store all the words in document.
     */
    var start = System.nanoTime()
    // Load and parse the data
    val file = sc.textFile(inputPath)
    val doc = file.map { line =>
      val w = line.trim.split(";;;;;", 2)
      val docno = w(0).trim.toLong
      val words = w(1).trim.split(" ")
      docno -> words
    }
    doc.persist(StorageLevel.MEMORY_AND_DISK)
    var end = System.nanoTime()
    val timeFileInput = end - start

    /**
     * We build a very large matrix to store the word frequence, the rows is the count
     * of documents, the cols is 2<<30. So large cols basically solved the hashing conflicting.
     * To save space of matrix, we represent it as sparse way. Our aim is trans rdd
     * doc:RDD[(Long, Array[String])] to hashDocWord:RDD[(Long, Vector)].
     * For each document in doc:RDD[(Long, Array[String])], we hash the word to a particular
     * column colIndex, and let Vector(colIndex)+=1. So, when one document has be precessing finished,
     * the document's word count information will stored in Vector.
     * Similarly, we present Vector as sparse way.
     */
    start = System.nanoTime()
    val hashTF = new HashingTF(1 << 30)
    val hashDocWord = hashTF.transform(doc.mapValues(_.toIterable))
    end = System.nanoTime()
    val timeHashTFConstruct = end - start

    /**
     * To construct a word column, we use word count algorithm to produce a sorted
     * word array with count value, it's totalWordCount:RDD[(String, Long)].
     * But to improve the performance of lda algorithm, we should reduce the size of word column by filtering
     * the words which count value is less than minDocFreq, result is reducedWordCount:RDD[(String, Long)].
     */
    start = System.nanoTime()
    val totalWordCount = doc.map(_._2).flatMap(_.toIterator).map(w => (w, 1L)).reduceByKey(_ + _)
    totalWordCount.persist(StorageLevel.MEMORY_AND_DISK)
    //save to hdfs
    totalWordCount.saveAsTextFile(outputPath + "/TotalWordCount")

    val reducedWordCount = totalWordCount.filter(_._2 > minDocFreq)
    reducedWordCount.persist(StorageLevel.MEMORY_AND_DISK)
    //save to hdfs
    reducedWordCount.saveAsTextFile(outputPath + "/ReducedWordCount")
    end = System.nanoTime()
    val timeWordCount = end - start

    /**
     * The input format of MLlib lda is RDD[(Long, Vector)].
     * Long postion store the docno, Vector store the frequence of words in document distributed by
     * word reduced column.
     * To achieve this, we first trans reduced word column to a index array corresponding
     * to the word in hash matrix.
     * Basing the index array, we trans hashDocWord to lda input format, the docWord:RDD[(Long, Vector)].
     */
    start = System.nanoTime()
    val wordColumn = reducedWordCount.map(_._1)
    //reduced word's index of word index in hashDocWord
    val indexArray = wordColumn.map(word => hashTF.indexOf(word)).collect()
    val bcIndexArray = hashDocWord.context.broadcast(indexArray)

    val docWord = hashDocWord.mapValues { tf =>
      val thisIndexArray = bcIndexArray.value
      var indexFreqArray = new ArrayBuffer[(Int, Double)]()
      for (i <- Range(0, thisIndexArray.length)) {
        val tfFreq = tf.apply(thisIndexArray(i))
        //sparse vector, filter tfFreq=0.0 elements, need tfFreq >= 1.0
        if (tfFreq > 0.5)
          indexFreqArray += (i -> tfFreq)
      }
      Vectors.sparse(thisIndexArray.length, indexFreqArray)
    }
    docWord.persist(StorageLevel.MEMORY_AND_DISK)
    //save to hdfs
    docWord.saveAsTextFile(outputPath + "/DocWordMatrix")
    end = System.nanoTime()
    val timeDocWordConstruct = end - start

    //    reducedWordCount.unpersist()
    //    totalWordCount.unpersist()
    //    hashDocWord.unpersist()
    //    doc.unpersist()

    // Cluster the documents into three topics using LDA
    start = System.nanoTime()
    val distributedLdaModel: DistributedLDAModel = new LDA()
      .setK(topicNum)
      .setAlpha(alpha)
      .setBeta(beta)
      .setMaxIterations(maxIterations)
      .setOptimizer(new EMLDAOptimizer)
      .run(docWord).asInstanceOf[DistributedLDAModel]
    end = System.nanoTime()
    val timeLdaRun = end - start

    /**
     * Get the result of lda, document distributed by topic, result format
     * is example as: doc t1 t2 t3 ... tn, t is the score of each topic possess.
     */
    start = System.nanoTime()
    val topicDistribution = distributedLdaModel.topicDistributions
    //save to hdfs
    topicDistribution.saveAsTextFile(outputPath + "/TopicDistribution")
    end = System.nanoTime()
    val timeTopicDistributionOutput = end - start

    /**
     * Output the word-topics matrix.
     * This is unrecommended when the input document is very more. Because under this way, the word-topics matrix
     * will be very large. The matrix is represent in dense way, and the matrix can't be stored distributed.
     * It is just a matrix on single node, not a spark rdd.
     */
    start = System.nanoTime()
    /*    val wordTopicMatrix = distributedLdaModel.topicsMatrix
        val wordTopicArray = wordTopicMatrix.toString(wordTopicMatrix.numRows, wordTopicMatrix.numCols * 24).split("\n")
        val wordTopic = sc.parallelize(wordTopicArray, 2)
        //save to hdfs
        wordTopic.saveAsTextFile(outputPath + "/WordTopicMatrix")*/
    end = System.nanoTime()
    val timeWordTopicMatrixOutput = end - start

    //spark context stop
    sc.stop()


    // one minute = how much nano?
    val minute = (BigInt(10).pow(9) * 60).toDouble
    println("------------------------------------------------------------------------------------")
    println("Unit: Minute")
    println(s"Total time: ${
      (timeFileInput + timeHashTFConstruct + timeWordCount + timeDocWordConstruct +
        timeLdaRun + timeTopicDistributionOutput + timeWordTopicMatrixOutput) / minute
    }")
    println("FileInput: " + timeFileInput / minute)
    println("HashTFConstruct: " + timeHashTFConstruct / minute)
    println("WordCount: " + timeWordCount / minute)
    println("DocWordConstruct: " + timeDocWordConstruct / minute)
    println("LdaRun: " + timeLdaRun / minute)
    println("TopicDistributionOutput: " + timeTopicDistributionOutput / minute)
    println("WordTopicMatrixOutput: " + timeWordTopicMatrixOutput / minute)
    println("------------------------------------------------------------------------------------")
  }
}
