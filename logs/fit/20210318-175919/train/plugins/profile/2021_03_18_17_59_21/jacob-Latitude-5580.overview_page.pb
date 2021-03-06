?	?9?S?%E@?9?S?%E@!?9?S?%E@	?) *???) *??!?) *??"w
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
	?~?n??C@?~?n??C@!?~?n??C@      ??!       "	B??????B??????!B??????*      ??!       2	Q??ڦx??Q??ڦx??!Q??ڦx??:	??^??????^????!??^????B      ??!       J	 9a?hV?? 9a?hV??! 9a?hV??R      ??!       Z	 9a?hV?? 9a?hV??! 9a?hV??b      ??!       JGPUY?) *??b qH?ۛ?yX@ypZ?Q???"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits??T?9d??!??T?9d??"-
IteratorGetNext/_1_Send??-m]???!?uA?????"C
%gradient_tape/sequential/dense/MatMulMatMul?k???z??!ȕ?^??0"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam??,????!,??̝??"5
sequential/dense/MatMulMatMulp?? E??!??B????0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul??݉??!q?b????"E
'gradient_tape/sequential/dense_1/MatMulMatMulQ;?%???!&mc:/8??0"7
sequential/dense_1/SoftmaxSoftmax?[???t??!?r{?y???"7
sequential/dense_1/MatMulMatMul???_،??!??y??#??0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad???C$???!!??ȉ???Q      Y@Y???cj`7@a??g?'S@q5o???;@y0?Hx????"?
both?Your program is POTENTIALLY input-bound because 94.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?27.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 