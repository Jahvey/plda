package com.tang

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.{SparseMatrix, DenseMatrix}

/**
 * Created by Tang Lizhe on 2015/9/12.
 * Just for testing some functions.
 */

object Test {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Test").setMaster("local")
    val sc = new SparkContext(conf)

    val r0 = sc.parallelize(1 to 10, 2)
    println(r0.collect())
  }
}
