import os

import tensorflow as tf

from utils.logger import get_logger

LOG = get_logger('trainer')


class UnetTrainer:

    def __init__(self, model, input, loss_fn, optimizer, metric, epoches):
        self.model = model
        self.input = input
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric = metric
        self.epoches = epoches

        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, './tf_ckpts', max_to_keep=3)

        self.train_log_dir = 'logs/gradient_tape/'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)

        self.model_save_path = 'saved_models/'

    def train_step(self, batch):
        trainable_variables = self.model.trainable_variables
        inputs, labels = batch
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)
            step_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(step_loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        self.metric.update_state(labels, predictions)

        return step_loss, predictions

    def train(self):
        for epoch in range(self.epoches):
            LOG.info(f'Start epoch {epoch}')

            step_loss = 0
            for step, training_batch in enumerate(self.input):
                step_loss, predictions = self.train_step(training_batch)
                LOG.info("Loss at step %d: %.2f" % (step, step_loss))

            train_acc = self.metric.result()
            LOG.info("Training acc over epoch: %.4f" % (float(train_acc)))

            save_path = self.checkpoint_manager.save()
            LOG.info("Saved checkpoint: {}".format(save_path))

            self._write_summary(step_loss, epoch)

            self.metric.reset_states()

        save_path = os.path.join(self.model_save_path, "unet/1/")
        tf.saved_model.save(self.model, save_path)

    def _write_summary(self, loss, epoch):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=epoch)
            tf.summary.scalar('accuracy', self.metric.result(), step=epoch)
            # tensorboard --logdir logs/gradient_tape
