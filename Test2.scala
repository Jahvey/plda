package com.tang

import org.apache.spark.mllib.linalg._


/**
 * Created by Tang Lizhe on 2015/9/29.
 */
object Test2 {
  def main(args: Array[String]): Unit = {
    val v = Vectors.sparse(10, Array(1,3,5,7), Array(1.0, 3.0, 5.0, 0.0))
    println(v.numActives)
    println(v.numNonzeros)
  }
}
