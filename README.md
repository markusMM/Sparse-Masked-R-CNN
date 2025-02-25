# Sparse-Masked-R-CNN
 An example od a sparse truncated Masked-R-CNN for image segmentation.

It is adapting the old Keras implementation of Mask R-CNNs and sparsified the loss functions.

For instance using L2 norms of the weights divided by their variance (if desired).
Each network, RPN, FPN, masking and the classifier can be included with such L1 and/or L2 norm.
The idea is similar to ridge or lasso. - I shall more or less avoid specific weights from taking too high numbers!

NOTE: This is a very old repository (2019) more or less restaurated!
