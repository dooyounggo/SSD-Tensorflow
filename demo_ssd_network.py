import os
import cv2
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as emtree
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization
from datasets.pascalvoc_to_tfrecords import VOC_LABELS


tf.app.flags.DEFINE_float(
    'select_threshold', 0.25, 'Selection threshold.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.45, 'Non-Maximum Selection threshold.')
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_integer(
    'max_samples', 500, 'The maximum number of samples.')
tf.app.flags.DEFINE_string(
    'dataset_dir', './demo/', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', './checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
tf.app.flags.DEFINE_string(
    'eval_dir', './logs/', 'Directory where the results are saved to.')
tf.app.flags.DEFINE_string(
    'save_gt', False, 'Whether to save ground truth detection results.')

FLAGS = tf.app.flags.FLAGS


def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=FLAGS.num_classes, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


def process_label(annotation_path):
    # Read the XML annotation file.
    tree = emtree.parse(annotation_path)
    root = tree.getroot()

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))
    return shape, bboxes, labels, labels_text, difficult, truncated


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    slim = tf.contrib.slim
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    isess = tf.InteractiveSession(config=config)

    # Input placeholder.
    net_shape = (300, 300)
    data_format = 'NHWC'
    img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
    # Evaluation pre-processing: resize to SSD net shape.
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)

    # Define the SSD model.
    reuse = True if 'ssd_net' in locals() else None
    ssd_net = ssd_vgg_300.SSDNet()
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

    # Restore SSD model.
    ckpt_filename = FLAGS.checkpoint_path
    # ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
    isess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(isess, ckpt_filename)

    # SSD default anchor boxes.
    ssd_anchors = ssd_net.anchors(net_shape)

    # Test on some demo image and visualize output.
    path = FLAGS.dataset_dir
    dataset_root = os.sep.join(path.split(os.sep)[:-1])
    image_names = sorted(os.listdir(path))
    for i, name in enumerate(image_names):
        if i >= FLAGS.max_samples:
            break
        if i % 50 == 0:
            print('Evaluating images... {:5d}/{}'.format(i, min(len(image_names), FLAGS.max_samples)))
        img = cv2.cvtColor(cv2.imread(os.path.join(path, name)), cv2.COLOR_BGR2RGB)
        rclasses, rscores, rbboxes = process_image(img, select_threshold=FLAGS.select_threshold,
                                                   nms_threshold=FLAGS.nms_threshold)
        visualization.plt_bboxes(img, rclasses, rscores, rbboxes, num_classes=FLAGS.num_classes,
                                 savefig_name=os.path.join(FLAGS.eval_dir, 'demo', f'{i:05d}.jpg'))

        if FLAGS.save_gt:
            annotation_path = os.path.join(dataset_root, 'Annotations', name.replace('.jpg', '.xml'))
            _, bboxes, _, labels, _, _ = process_label(annotation_path)
            scores = np.ones_like(labels, dtype=np.float32)
            visualization.plt_bboxes(img, labels, scores, bboxes, num_classes=FLAGS.num_classes,
                                     savefig_name=os.path.join(FLAGS.eval_dir, 'demo_gt', f'{i:05d}.jpg'))
