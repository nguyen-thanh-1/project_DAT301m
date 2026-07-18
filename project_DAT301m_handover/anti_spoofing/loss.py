import tensorflow as tf

def focal_loss_from_logits(labels, logits, alpha_weights, gamma=2.0, label_smoothing=0.0, num_classes=2):
    labels_1h = tf.one_hot(labels, num_classes)
    if label_smoothing > 0:
        labels_1h = labels_1h * (1 - label_smoothing) + label_smoothing / num_classes

    probs = tf.nn.softmax(tf.cast(logits, tf.float32), axis=-1)
    probs = tf.clip_by_value(probs, 1e-7, 1.0 - 1e-7)

    ce = -labels_1h * tf.math.log(probs)
    focal_weight = tf.pow(1.0 - probs, gamma)
    alpha = tf.constant([alpha_weights[0], alpha_weights[1]], dtype=tf.float32)
    loss = alpha * focal_weight * ce
    loss = tf.reduce_sum(loss, axis=-1)
    return tf.reduce_mean(loss)
