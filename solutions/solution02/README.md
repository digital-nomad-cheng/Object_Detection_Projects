1. Faster RCNN has higher confidence for objects.
2. Results from evaluate_coco.py
    + RetinaNet
        ```buildoutcfg
        Loading and preparing results...
        DONE (t=0.69s)
        creating index...
        index created!
        Running per image evaluation...
        Evaluate annotation type *bbox*
        COCOeval_opt.evaluate() finished in 8.84 seconds.
        Accumulating evaluation results...
        COCOeval_opt.accumulate() finished in 1.37 seconds.
         Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.374
         Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.567
         Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.403
         Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.231
         Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.416
         Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.483
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.319
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.518
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.551
         Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.372
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.591
         Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.701
        ```
    + Faster RCNN
        ```buildoutcfg
        Loading and preparing results...
        DONE (t=0.36s)
        creating index...
        index created!
        Running per image evaluation...
        Evaluate annotation type *bbox*
        COCOeval_opt.evaluate() finished in 5.73 seconds.
        Accumulating evaluation results...
        COCOeval_opt.accumulate() finished in 0.77 seconds.
         Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.379
         Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.588
         Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.410
         Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.224
         Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.411
         Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.491
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.315
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.499
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.524
         Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.342
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.557
         Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.651
        ```