# Digit Recognizer Lab Notebook

## 2020-05-04

* Define the objective in business terms.
  * The objective is to classify as many digits correctly as possible.
* How will your solution be used?
  * This is not stated in the problem.
* What are the current solutions/workarounds (if any)?
  * Presumably the current solution is a manual one, and we are looking to automate it.
* How should you frame this problem (supervised/unsupervised, online/offline, etc.)?
  * This is an offline supervised learning problem.
* How should performance be measured?
  * Performance will be measured by categorization accuracy.
* Is the performance measure aligned with the business objective?
  * Not sure, as the business objective is not stated.
* What would be the minimum performance needed to reach the business objective?
  * I think above 90% would be pretty good.  And above 97% would be very good. http://yann.lecun.com/exdb/mnist/
* What are comparable problems?  Can you reuse experience or tools?
  * I haven't solved any comparable problems so far.
* Is human expertise available?
  * Yes, I am an expert (we all are) in handwriting recognition.
* How would you solve the problem manually?
  * I would solve this visually, not looking at the matrix of data.  This indicates to me that the current matrix may not be the best way to represent the data.  I may need to transform it before I train a model.
* List the assumptions you (or others) have made so far.
  * The data is labeled correctly.
  * The numbers are relatively centered.
* Verify assumptions if possible.