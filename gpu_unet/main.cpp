#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <utility>
#include <string>
#include <cstring>
#include <cmath>
#include <cuda.h>

extern "C" {
#ifdef __linux__
#include <unistd.h>
#else
#include <io.h>
#define F_OK    0
#define access _access
#endif
}

#include "./tensor.h"
#include "./activations.h"
#include "./conv.h"
#include "./optimizer.h"
#include "./loss.h"
#include "./pooling.h"
#include "./dropout.h"
#include "./utils.h"
#include "./data.h"

#include "./conv-relu.h"
#include "./conv-softmax.h"
#include "./unet-up.h"
#include "./unet-down.h"

static float sum(const Tensor<float, 2> &t) {
	t.moveToHost();
	double sum = 0.;
	for (int x = 0; x < t.dim(0); x++)
		for (int y = 0; y < t.dim(1); y++)
			sum += t(x, y);

	return sum;
}

static std::pair<Tensor<float, 4>, Tensor<float, 4>> testInput(int batch_size, int image_size) {
	Tensor<float, 4> images(batch_size, 1, image_size, image_size);
	Tensor<float, 4> truths(batch_size, 2, image_size, image_size);

	srand(1234);

	for (int b = 0; b < batch_size; b++) {
		for (int x = 0; x < image_size; x++) {
			for (int y = 0; y < image_size; y++) {
				bool t = (rand() & 0b011) == 0;
				images(b, 0, x, y) = t ? -0.5 : 0.5;
				truths(b, 0, x, y) = t ? 1. : 0.;
				truths(b, 1, x, y) = t ? 0. : 1.;
			}
		}
	}

	images.moveToDevice();
	truths.moveToDevice();
	return { images, truths };
}

static void bench_conv(int iterations) {
	float learning_rate = 1e-6;
	int batch_size = 5,
	    image_size = 256;

	ConvReLU conv1(1, 32, 3);
	ConvSoftMax conv2(32, 2);

	PerPixelCELoss loss;
	
	conv1.optimizer = new Sgd(learning_rate);
	conv2.optimizer = new Sgd(learning_rate);
	
	// Inizializzazione migliorata
	float init1 = sqrt(2.0 / (3 * 3 * 1));
	float init2 = sqrt(2.0 / (1 * 1 * 32));
	uniformRandomInit(-init1, init1, conv1.weights, conv1.bias);
	uniformRandomInit(-init2, init2, conv2.weights, conv2.bias);

	float elapsed,
	      elapsed_time_forward = 0.,
	      elapsed_time_backward = 0.;
	cudaEvent_t start, stop;
	cudaErrchk(cudaEventCreate(&start));
	cudaErrchk(cudaEventCreate(&stop));

	auto data = testInput(batch_size, image_size);
	
	for (int i = 0; i < iterations; i++) {
		cudaErrchk(cudaEventRecord(start, 0));
		auto t1 = conv1.forward(data.first);
		cudaErrchk(cudaEventRecord(stop, 0));
		cudaErrchk(cudaEventSynchronize(stop));
		cudaErrchk(cudaEventElapsedTime(&elapsed, start, stop));
		elapsed_time_forward += elapsed;

		cudaErrchk(cudaEventRecord(start, 0));
		auto pred = conv2.forward(t1);
		cudaErrchk(cudaEventRecord(stop, 0));
		cudaErrchk(cudaEventSynchronize(stop));
		cudaErrchk(cudaEventElapsedTime(&elapsed, start, stop));
		elapsed_time_forward += elapsed;

		if (i % 10 == 0) {
			auto l = loss.forward(pred, data.second);
			float loss_val = sum(l);
			printf("iter. %d:\tloss=%f\n", i, loss_val);
			
			if (std::isnan(loss_val) || std::isinf(loss_val)) {
				fprintf(stderr, "ERROR: Loss is NaN/Inf at iteration %d!\n", i);
				break;
			}
		}

		auto e1 = loss.backward(pred, data.second);

		cudaErrchk(cudaEventRecord(start, 0));
		auto e2 = conv2.backward(e1);
		cudaErrchk(cudaEventRecord(stop, 0));
		cudaErrchk(cudaEventSynchronize(stop));
		cudaErrchk(cudaEventElapsedTime(&elapsed, start, stop));
		elapsed_time_backward += elapsed;

		cudaErrchk(cudaEventRecord(start, 0));
		auto e3 = conv1.backward(e2);
		cudaErrchk(cudaEventRecord(stop, 0));
		cudaErrchk(cudaEventSynchronize(stop));
		cudaErrchk(cudaEventElapsedTime(&elapsed, start, stop));
		elapsed_time_backward += elapsed;
	}

	cudaErrchk(cudaEventDestroy(start));
	cudaErrchk(cudaEventDestroy(stop));
	printf("conv. forward:  %.3fms\n", elapsed_time_forward);
	printf("conv. backward: %.3fms\n", elapsed_time_backward);
}

static void bench_mini_unet(int iterations) {
	int batch_size = 1;
	float learning_rate = 1e-5;
	float elapsed,
	      elapsed_conv_forward = 0.,
	      elapsed_conv_backward = 0.,
	      elapsed_down_forward = 0.,
	      elapsed_down_backward = 0.,
	      elapsed_up_forward = 0.,
	      elapsed_up_backward = 0.;

	MembraneLoader dataloader("../data/cell-membranes/", batch_size);

	cudaEvent_t start, stop;
	cudaErrchk(cudaEventCreate(&start));
	cudaErrchk(cudaEventCreate(&stop));

	auto time_start = [&]() -> void {
		cudaErrchk(cudaPeekAtLastError());
		cudaErrchk(cudaEventRecord(start, 0));
	};

	auto time_end = [&](float &t) -> void {
		cudaErrchk(cudaEventRecord(stop, 0));
		cudaErrchk(cudaEventSynchronize(stop));
		cudaErrchk(cudaEventElapsedTime(&elapsed, start, stop));
		cudaErrchk(cudaDeviceSynchronize());
		t += elapsed;
	};

	ConvReLU conv1(1, 20, 3);
	UnetDown down1(2);

	ConvReLU conv2(20, 40, 3);
	UnetDown down2(2);

	ConvReLU conv3(40, 80, 3);
	UnetUp up3(2, conv2.output_channels, conv3.output_channels);

	ConvReLU conv4(80 + 40, 80, 3);
	UnetUp up4(2, conv1.output_channels, conv4.output_channels);

	ConvSoftMax conv5(80 + 20, 2);

	PerPixelCELoss loss;
	
	// Inizializzazione He per ReLU
	float init1 = sqrt(2.0 / (3 * 3 * 1));
	float init2 = sqrt(2.0 / (3 * 3 * 20));
	float init3 = sqrt(2.0 / (3 * 3 * 40));
	float init4 = sqrt(2.0 / (3 * 3 * 120));
	float init5 = sqrt(2.0 / (1 * 1 * 100));
	
	conv1.optimizer = new Sgd(learning_rate);
	conv2.optimizer = new Sgd(learning_rate);
	conv3.optimizer = new Sgd(learning_rate);
	conv4.optimizer = new Sgd(learning_rate);
	conv5.optimizer = new Sgd(learning_rate);
	
	uniformRandomInit(-init1, init1, conv1.weights, conv1.bias);
	uniformRandomInit(-init2, init2, conv2.weights, conv2.bias);
	uniformRandomInit(-init3, init3, conv3.weights, conv3.bias);
	uniformRandomInit(-init4, init4, conv4.weights, conv4.bias);
	uniformRandomInit(-init5, init5, conv5.weights, conv5.bias);

	for (int iter = 0; iter < iterations; iter++) {
		auto data = dataloader.loadBatch();

		time_start();
		auto forward_conv1 = conv1.forward(data.first);
		time_end(elapsed_conv_forward);
		time_start();
		auto forward_down1 = down1.forward(forward_conv1);
		time_end(elapsed_down_forward);

		time_start();
		auto forward_conv2 = conv2.forward(forward_down1);
		time_end(elapsed_conv_forward);
		time_start();
		auto forward_down2 = down2.forward(forward_conv2);
		time_end(elapsed_down_forward);

		time_start();
		auto forward_conv3 = conv3.forward(forward_down2);
		time_end(elapsed_conv_forward);
		time_start();
		auto forward_up3 = up3.forward(forward_conv2, forward_conv3);
		time_end(elapsed_up_forward);

		time_start();
		auto forward_conv4 = conv4.forward(forward_up3);
		time_end(elapsed_conv_forward);
		time_start();
		auto forward_up4 = up4.forward(forward_conv1, forward_conv4);
		time_end(elapsed_up_forward);

		time_start();
		auto forward_conv5 = conv5.forward(forward_up4);
		time_end(elapsed_conv_forward);

		if (iter % 10 == 0) {
			auto l = loss.forward(forward_conv5, data.second);
			float loss_val = sum(l);
			printf("iter. %d:\tloss=%f\n", iter, loss_val);
			
			if (std::isnan(loss_val) || std::isinf(loss_val)) {
				fprintf(stderr, "ERROR: Loss is NaN/Inf at iteration %d!\n", iter);
				break;
			}
		}

		auto backward_loss = loss.backward(forward_conv5, data.second);
		time_start();
		auto backward_conv5 = conv5.backward(backward_loss);
		time_end(elapsed_conv_backward);

		time_start();
		auto backward_up4 = up4.backward(backward_conv5);
		time_end(elapsed_up_backward);
		time_start();
		auto backward_conv4 = conv4.backward(backward_up4);
		time_end(elapsed_conv_backward);

		time_start();
		auto backward_up3 = up3.backward(backward_conv4);
		time_end(elapsed_up_backward);
		time_start();
		auto backward_conv3 = conv3.backward(backward_up3);
		time_end(elapsed_conv_backward);

		time_start();
		auto backward_down2 = down2.backward(backward_conv3, backward_conv4);
		time_end(elapsed_down_backward);
		time_start();
		auto backward_conv2 = conv2.backward(backward_down2);
		time_end(elapsed_conv_backward);

		time_start();
		auto backward_down1 = down1.backward(backward_conv2, backward_conv5);
		time_end(elapsed_down_backward);
		time_start();
		auto backward_conv1 = conv1.backward(backward_down1);
		time_end(elapsed_conv_backward);
	}

	cudaErrchk(cudaDeviceSynchronize());
	cudaErrchk(cudaEventDestroy(start));
	cudaErrchk(cudaEventDestroy(stop));
	printf("convolution forward:   %.3fms\n", elapsed_conv_forward / 5.);
	printf("convolution backward:  %.3fms\n", elapsed_conv_backward / 5.);
	printf("down-layer forward:    %.3fms\n", elapsed_down_forward / 2.);
	printf("down-layer backward:   %.3fms\n", elapsed_down_backward / 2.);
	printf("up-layer forward:      %.3fms\n", elapsed_up_forward / 2.);
	printf("up-layer backward:     %.3fms\n", elapsed_up_backward / 2.);
}

static void bench_mnist(int iterations) {
	int batch_size = 15;
	float learning_rate = 1e-5;

	/* 32x32 */
	ConvReLU conv1(1, 16, 3);
	MaxPool pool1(2);

	/* 16x16 */
	ConvReLU conv2(16, 32, 3);
	MaxPool pool2(2);

	/* 8x8 */
	ConvReLU conv3(32, 64, 3);
	MaxPool pool3(8);

	/*1x1*/
	ConvSoftMax conv4(64, 10);

	PerPixelCELoss loss;
	
	conv1.optimizer = new Sgd(learning_rate);
	conv2.optimizer = new Sgd(learning_rate);
	conv3.optimizer = new Sgd(learning_rate);
	conv4.optimizer = new Sgd(learning_rate);
	
	// Inizializzazione He
	float init1 = sqrt(2.0 / (3 * 3 * 1));
	float init2 = sqrt(2.0 / (3 * 3 * 16));
	float init3 = sqrt(2.0 / (3 * 3 * 32));
	float init4 = sqrt(2.0 / (1 * 1 * 64));
	uniformRandomInit(-init1, init1, conv1.weights, conv1.bias);
	uniformRandomInit(-init2, init2, conv2.weights, conv2.bias);
	uniformRandomInit(-init3, init3, conv3.weights, conv3.bias);
	uniformRandomInit(-init4, init4, conv4.weights, conv4.bias);

	MNISTLoader mnist("../data/mnist/mnist-train.txt", batch_size);
	
	for (int iter = 0; iter < iterations; iter++) {
		auto data = mnist.loadBatch();

		auto t1 = pool1.forward(conv1.forward(data.first));
		auto t2 = pool2.forward(conv2.forward(t1));
		auto t3 = pool3.forward(conv3.forward(t2));
		auto pred = conv4.forward(t3);

		if (iter % 20 == 0) {
			float l = sum(loss.forward(pred, data.second));
			float acc = mnist.checkAccuracy(pred, data.second);
			printf("iter. %d:\tloss=%f, correct_digits=%.1f%%\n", iter, l, acc);
			
			if (std::isnan(l) || std::isinf(l)) {
				fprintf(stderr, "ERROR: Loss is NaN/Inf at iteration %d!\n", iter);
				break;
			}
		}

		auto e = loss.backward(pred, data.second);
		auto e4 = conv4.backward(e);
		auto e3 = conv3.backward(pool3.backward(e4));
		auto e2 = conv2.backward(pool2.backward(e3));
		auto e1 = conv1.backward(pool1.backward(e2));
	}
}

static void test_unet(int iterations) {
	int batch_size = 2;
	float learning_rate = 1e-6;  // Ridotto ulteriormente
	
	ConvReLU conv_0_1(1, 2, 3);
	ConvReLU conv_0_2(2, 4, 3);
	MaxPool pool_0(2);

	ConvReLU conv_1_1(4, 8, 3);
	MaxPool pool_1(2);
	Dropout drop_0;

	ConvReLU conv_2_1(8, 16, 3);
	MaxPool pool_2(2);
	Dropout drop_1;

	ConvReLU conv_3_1(16, 16, 3);
	ConvReLU conv_3_2(16, 8, 3);
	Upsample upsample_3(2);

	ConvReLU conv_4_1(8 + 16, 8, 3);
	Upsample upsample_4(2);

	ConvReLU conv_5_1(8 + 8, 8, 3);
	Upsample upsample_5(2);

	ConvReLU conv_6_1(8 + 4, 4, 3);
	ConvSoftMax conv_6_2(4, 2);

	PerPixelCELoss loss;

	if (access("./weights/unet/conv-0-1.weights", F_OK) != 0) {
		// Inizializzazione He migliorata
		float init_0_1 = sqrt(2.0 / (3 * 3 * 1));
		float init_0_2 = sqrt(2.0 / (3 * 3 * 2));
		float init_1_1 = sqrt(2.0 / (3 * 3 * 4));
		float init_2_1 = sqrt(2.0 / (3 * 3 * 8));
		float init_3_1 = sqrt(2.0 / (3 * 3 * 16));
		float init_3_2 = sqrt(2.0 / (3 * 3 * 16));
		float init_4_1 = sqrt(2.0 / (3 * 3 * 24));
		float init_5_1 = sqrt(2.0 / (3 * 3 * 16));
		float init_6_1 = sqrt(2.0 / (3 * 3 * 12));
		float init_6_2 = sqrt(2.0 / (1 * 1 * 4));
		
		printf("Inizializzazione pesi con He initialization...\n");
		uniformRandomInit(-init_0_1, init_0_1, conv_0_1.weights, conv_0_1.bias);
		uniformRandomInit(-init_0_2, init_0_2, conv_0_2.weights, conv_0_2.bias);
		uniformRandomInit(-init_1_1, init_1_1, conv_1_1.weights, conv_1_1.bias);
		uniformRandomInit(-init_2_1, init_2_1, conv_2_1.weights, conv_2_1.bias);
		uniformRandomInit(-init_3_1, init_3_1, conv_3_1.weights, conv_3_1.bias);
		uniformRandomInit(-init_3_2, init_3_2, conv_3_2.weights, conv_3_2.bias);
		uniformRandomInit(-init_4_1, init_4_1, conv_4_1.weights, conv_4_1.bias);
		uniformRandomInit(-init_5_1, init_5_1, conv_5_1.weights, conv_5_1.bias);
		uniformRandomInit(-init_6_1, init_6_1, conv_6_1.weights, conv_6_1.bias);
		uniformRandomInit(-init_6_2, init_6_2, conv_6_2.weights, conv_6_2.bias);
	} else {
		printf("Caricamento pesi da file...\n");
		readFromFile("./weights/unet/conv-0-1.weights", conv_0_1.weights, conv_0_1.bias);
		readFromFile("./weights/unet/conv-0-2.weights", conv_0_2.weights, conv_0_2.bias);
		readFromFile("./weights/unet/conv-1-1.weights", conv_1_1.weights, conv_1_1.bias);
		readFromFile("./weights/unet/conv-2-1.weights", conv_2_1.weights, conv_2_1.bias);
		readFromFile("./weights/unet/conv-3-1.weights", conv_3_1.weights, conv_3_1.bias);
		readFromFile("./weights/unet/conv-3-2.weights", conv_3_2.weights, conv_3_2.bias);
		readFromFile("./weights/unet/conv-4-1.weights", conv_4_1.weights, conv_4_1.bias);
		readFromFile("./weights/unet/conv-5-1.weights", conv_5_1.weights, conv_5_1.bias);
		readFromFile("./weights/unet/conv-6-1.weights", conv_6_1.weights, conv_6_1.bias);
		readFromFile("./weights/unet/conv-6-2.weights", conv_6_2.weights, conv_6_2.bias);
	}

	MembraneLoader dataloader("../data/cell-membranes/", batch_size);

	conv_0_1.optimizer = new Sgd(learning_rate);
	conv_0_2.optimizer = new Sgd(learning_rate);
	conv_1_1.optimizer = new Sgd(learning_rate);
	conv_2_1.optimizer = new Sgd(learning_rate);
	conv_3_1.optimizer = new Sgd(learning_rate);
	conv_3_2.optimizer = new Sgd(learning_rate);
	conv_4_1.optimizer = new Sgd(learning_rate);
	conv_5_1.optimizer = new Sgd(learning_rate);
	conv_6_1.optimizer = new Sgd(learning_rate);
	conv_6_2.optimizer = new Sgd(learning_rate);

	float best_loss = 1e9;
	int no_improvement_count = 0;
	
	for (int iter = 0; iter < iterations; iter++) {
		auto data = dataloader.loadBatch();

		/***			Forward				***/
		auto t0 = conv_0_2.forward(conv_0_1.forward(data.first));
		auto t1 = conv_1_1.forward(pool_0.forward(t0));
		auto t2 = conv_2_1.forward(drop_0.forward(pool_1.forward(t1)));
		auto t3 = conv_3_2.forward(conv_3_1.forward(drop_1.forward(pool_2.forward(t2))));
		auto t4in = concat(upsample_3.forward(t3), t2);
		auto t4 = conv_4_1.forward(t4in);
		auto t5in = concat(upsample_4.forward(t4), t1);
		auto t5 = conv_5_1.forward(t5in);
		auto t6in = concat(upsample_5.forward(t5), t0);
		auto pred = conv_6_2.forward(conv_6_1.forward(t6in));

		float l = sum(loss.forward(pred, data.second));

		/***			Backward			***/
		auto e6 = conv_6_1.backward(conv_6_2.backward(loss.backward(pred, data.second)));
		auto e6split = split(e6, t5.dim(1));
		auto e5 = conv_5_1.backward(upsample_5.backward(e6split.first));
		auto e5split = split(e5, t4.dim(1));
		auto e4 = conv_4_1.backward(upsample_4.backward(e5split.first));
		auto e4split = split(e4, t3.dim(1));
		auto e3 = conv_3_1.backward(conv_3_2.backward(upsample_3.backward(e4split.first)));
		auto e2 = conv_2_1.backward(pool_2.backward(drop_1.backward(e3)) + e4split.second);
		auto e1 = conv_1_1.backward(pool_1.backward(drop_0.backward(e2)) + e5split.second);
		auto e0 = conv_0_1.backward(conv_0_2.backward(pool_0.backward(e1) + e6split.second));

		/***		Stampa Metriche			***/
		printf("iter. %d:\tloss=%.2f", iter, l);
		
		if (iter % 10 == 0) {
			float acc = dataloader.checkAccuracy(pred, data.second);
			printf("\tcorrect_pixels=%.2f%%", acc);
		}
		printf("\n");
		
		// Controllo NaN/Inf
		if (std::isnan(l) || std::isinf(l)) {
			fprintf(stderr, "ERRORE: Loss è NaN/Inf all'iterazione %d!\n", iter);
			fprintf(stderr, "Training interrotto.\n");
			break;
		}
		
		// Early stopping semplice
		if (l < best_loss) {
			best_loss = l;
			no_improvement_count = 0;
		} else {
			no_improvement_count++;
		}
		
		// Learning rate decay ogni 200 iterazioni
		if (iter > 0 && iter % 200 == 0) {
			learning_rate *= 0.95;
			printf("Learning rate ridotto a: %.6f\n", learning_rate);
			
			// Aggiorna tutti gli optimizer
			delete conv_0_1.optimizer; conv_0_1.optimizer = new Sgd(learning_rate);
			delete conv_0_2.optimizer; conv_0_2.optimizer = new Sgd(learning_rate);
			delete conv_1_1.optimizer; conv_1_1.optimizer = new Sgd(learning_rate);
			delete conv_2_1.optimizer; conv_2_1.optimizer = new Sgd(learning_rate);
			delete conv_3_1.optimizer; conv_3_1.optimizer = new Sgd(learning_rate);
			delete conv_3_2.optimizer; conv_3_2.optimizer = new Sgd(learning_rate);
			delete conv_4_1.optimizer; conv_4_1.optimizer = new Sgd(learning_rate);
			delete conv_5_1.optimizer; conv_5_1.optimizer = new Sgd(learning_rate);
			delete conv_6_1.optimizer; conv_6_1.optimizer = new Sgd(learning_rate);
			delete conv_6_2.optimizer; conv_6_2.optimizer = new Sgd(learning_rate);
		}
	}

	printf("Training completato! (Salvataggio pesi...)\n");
	writeToFile("./weights/unet/conv-0-1.weights", conv_0_1.weights, conv_0_1.bias);
	writeToFile("./weights/unet/conv-0-2.weights", conv_0_2.weights, conv_0_2.bias);
	writeToFile("./weights/unet/conv-1-1.weights", conv_1_1.weights, conv_1_1.bias);
	writeToFile("./weights/unet/conv-2-1.weights", conv_2_1.weights, conv_2_1.bias);
	writeToFile("./weights/unet/conv-3-1.weights", conv_3_1.weights, conv_3_1.bias);
	writeToFile("./weights/unet/conv-3-2.weights", conv_3_2.weights, conv_3_2.bias);
	writeToFile("./weights/unet/conv-4-1.weights", conv_4_1.weights, conv_4_1.bias);
	writeToFile("./weights/unet/conv-5-1.weights", conv_5_1.weights, conv_5_1.bias);
	writeToFile("./weights/unet/conv-6-1.weights", conv_6_1.weights, conv_6_1.bias);
	writeToFile("./weights/unet/conv-6-2.weights", conv_6_2.weights, conv_6_2.bias);
	printf("Pesi salvati con successo!\n");
}

int main(int argc, const char *argv[]) {
	cudaErrchk(cudaDeviceSynchronize());
	printf("Dispositivo CUDA sincronizzato!\n");
	
	if (argc == 1) {
		bench_mini_unet(1);
		return EXIT_SUCCESS;
	}

	if (argc != 3) {
		fprintf(stderr, "uso: %s <bench_unet|test_unet|mnist|bench_conv> <iterazioni>\n", argv[0]);
		return EXIT_FAILURE;
	}

	int iterations = atoi(argv[2]);
	
	if (strcmp(argv[1], "bench_unet") == 0)
		bench_mini_unet(iterations);
	else if (strcmp(argv[1], "test_unet") == 0)
		test_unet(iterations);
	else if (strcmp(argv[1], "mnist") == 0)
		bench_mnist(iterations);
	else if (strcmp(argv[1], "bench_conv") == 0)
		bench_conv(iterations);
	else {
		fprintf(stderr, "Comando non riconosciuto: %s\n", argv[1]);
		fprintf(stderr, "uso: %s <bench_unet|test_unet|mnist|bench_conv> <iterazioni>\n", argv[0]);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}