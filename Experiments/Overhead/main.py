import TensorQuant as tq
import tensorflow as tf
import time

def main():
    epochs = 80
    tensors = 1000
    k = 200

    fpq = tq.Quantize.Quantizers.FixedPointQuantizer_nearest(16,8)

    def make_variables(k, initializer):
        return tf.Variable(initializer(shape=[k, k], dtype=tf.float32))


    total_time = 0
    for i in range(0,tensors):
        w = make_variables(k, tf.random_normal_initializer(mean=1., stddev=2.))
        start = time.time()
        w_quantized = fpq.quantize(w)
        end = time.time()
        total_time += (end - start)

    print('Time for quantizing',tensors, 'shape', k,'x',k,'tensors with',epochs,'epochs',(total_time)*epochs)
    print(w)
    print(w_quantized)

    time_sequential = 1.0 #einmal quantize() sequentiell durchrechen lassen und messen
    time_n_gpus = 0.1 #einmal quantize()mit n = number of gpus  durchrechen lassen und messen
    speedup = time_n_gpus/time_sequential # speedup
    print('speedup', speedup)
if __name__ == "__main__":
    main()
