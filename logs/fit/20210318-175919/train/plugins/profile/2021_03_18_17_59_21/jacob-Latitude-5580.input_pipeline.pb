	?9?S?%E@?9?S?%E@!?9?S?%E@	?) *???) *??!?) *??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?9?S?%E@?~?n??C@1B??????AQ??ڦx??I??^????Y 9a?hV??*	????x?]@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatt&m????!t?3???=@)?Z?О?1??	??c9@:Preprocessing2F
Iterator::ModelYM?]??!?Vm&?3B@)?U?Z??1??%592@:Preprocessing2U
Iterator::Model::ParallelMapV2?D-ͭ??!???'1.2@)?D-ͭ??1???'1.2@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziph>?n?K??!???L?O@)&?fe???1SH\a?+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?B˺,??!?
NA?0@)?ڧ?1??1??J'P!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicem????U??!??/??@)m????U??1??/??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??;?2t?!???9?@)??;?2t?1???9?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap}??z?V??!??C??4@)ϣ????p?1b4??s@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 94.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?) *??IH?ۛ?yX@QpZ?Q??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?~?n??C@?~?n??C@!?~?n??C@      ??!       "	B??????B??????!B??????*      ??!       2	Q??ڦx??Q??ڦx??!Q??ڦx??:	??^??????^????!??^????B      ??!       J	 9a?hV?? 9a?hV??! 9a?hV??R      ??!       Z	 9a?hV?? 9a?hV??! 9a?hV??b      ??!       JGPUY?) *??b qH?ۛ?yX@ypZ?Q??