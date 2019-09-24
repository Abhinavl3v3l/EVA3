# EVA3
Extensive Vision AI Program 3.0
# Assignment 1B

### What are Channels and Kernels (according to EVA)?

1. Images are combination of pixels, pixels individually is just a dot/square of a certain ratio of RGB channel.

![Original Image](https://2.bp.blogspot.com/-tkggOOkYd00/Voln4EZhHrI/AAAAAAAAAVg/s81ocGoNFcc/s400/mandrill.png)

![RGB CHANNELS](https://4.bp.blogspot.com/-kAtLMhm4lCM/VolnvzNgtdI/AAAAAAAAAVY/EMESzQKQdYo/s640/mandrill_rgb.png)

2. Similar pixels together may form a patter, like part of a straight line or part of curve.

<img src="https://upload.wikimedia.org/wikipedia/commons/2/2b/Pixel-example.png" alt="Image Breakdown"  />

For example, looking at very zoomed in picture the edge of keyboard is a continuous collection of horizontal lines at a certain angle and the stand of monitor forms a curve line. 



#### Kernels and Channels

1. Hence an image can be broken down into set of features, and what detects these features are called **feature extractor**, **filters** or **kernels**.

2. Each kernel extract only one type of feature in an entire image and collection of similar features is called a **channel**.

   - Features of similar type, for example, all edges of $45^\circ$ or all edges of $90^\circ$  grouped together in respective channels using a $45^\circ$ edge detector or a vertical edge detector respectively.

   <img src="https://qph.fs.quoracdn.net/main-qimg-4bfdf63a4c5b24590f0deec9673eaee5-c" alt="Kernels"  />

   <img src="https://wiki.tum.de/download/thumbnails/23572254/filter%20levels.png?version=1&amp;modificationDate=1485348352200&amp;api=v2" style="zoom: 150%;" />

3. Combination of different channels combine to form complex features like a $\frac{1}{4} ^{th}$ of circle or  corner of a window, a cross or a plus sign.

> Complex things are made from small features. Similarly images can be imagined as combination ( in ascending order) of   features, gradients ,textures,  patterns, parts of objects, object and scenery 

------

### Why should we only (well mostly) use 3x3 Kernels?

- **Axis of Symmetry**  - Any Odd sized imaged can be achieved. Objective is to create a kernel which detects a feature.  Due of axis if symmetry any primitive feature can easily be created. Its hard to do the same with even sized kernel say 4x4. 

------

### How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)

No of Step to reach 1x1  = $99^{th}$ step

Calculation :

| Sno                                                          | Image size (x and y)                                         | Convolution                                                  | Resultant Image (x and y)                                    | Receptive Field                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1<br/>2<br/>3<br/>4<br/>5<br/>6<br/>7<br/>8<br/>9<br/>10<br/>11<br/>12<br/>13<br/>14<br/>15<br/>16<br/>17<br/>18<br/>19<br/>20<br/>21<br/>22<br/>23<br/>24<br/>25<br/>26<br/>27<br/>28<br/>29<br/>30<br/>31<br/>32<br/>33<br/>34<br/>35<br/>36<br/>37<br/>38<br/>39<br/>40<br/>41<br/>42<br/>43<br/>44<br/>45<br/>46<br/>47<br/>48<br/>49<br/>50<br/>51<br/>52<br/>53<br/>54<br/>55<br/>56<br/>57<br/>58<br/>59<br/>60<br/>61<br/>62<br/>63<br/>64<br/>65<br/>66<br/>67<br/>68<br/>69<br/>70<br/>71<br/>72<br/>73<br/>74<br/>75<br/>76<br/>77<br/>78<br/>79<br/>80<br/>81<br/>82<br/>83<br/>84<br/>85<br/>86<br/>87<br/>88<br/>89<br/>90<br/>91<br/>92<br/>93<br/>94<br/>95<br/>96<br/>97<br/>98<br/>99<br/>==100==<br/>101 | 199<br/>199<br/>197<br/>195<br/>193<br/>191<br/>189<br/>187<br/>185<br/>183<br/>181<br/>179<br/>177<br/>175<br/>173<br/>171<br/>169<br/>167<br/>165<br/>163<br/>161<br/>159<br/>157<br/>155<br/>153<br/>151<br/>149<br/>147<br/>145<br/>143<br/>141<br/>139<br/>137<br/>135<br/>133<br/>131<br/>129<br/>127<br/>125<br/>123<br/>121<br/>119<br/>117<br/>115<br/>113<br/>111<br/>109<br/>107<br/>105<br/>103<br/>101<br/>99<br/>97<br/>95<br/>93<br/>91<br/>89<br/>87<br/>85<br/>83<br/>81<br/>79<br/>77<br/>75<br/>73<br/>71<br/>69<br/>67<br/>65<br/>63<br/>61<br/>59<br/>57<br/>55<br/>53<br/>51<br/>49<br/>47<br/>45<br/>43<br/>41<br/>39<br/>37<br/>35<br/>33<br/>31<br/>29<br/>27<br/>25<br/>23<br/>21<br/>19<br/>17<br/>15<br/>13<br/>11<br/>9<br/>7<br/>5<br/>==3==<br/>1 | 3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>3x3<br/>==3x3== | 199<br/>197<br/>195<br/>193<br/>191<br/>189<br/>187<br/>185<br/>183<br/>181<br/>179<br/>177<br/>175<br/>173<br/>171<br/>169<br/>167<br/>165<br/>163<br/>161<br/>159<br/>157<br/>155<br/>153<br/>151<br/>149<br/>147<br/>145<br/>143<br/>141<br/>139<br/>137<br/>135<br/>133<br/>131<br/>129<br/>127<br/>125<br/>123<br/>121<br/>119<br/>117<br/>115<br/>113<br/>111<br/>109<br/>107<br/>105<br/>103<br/>101<br/>99<br/>97<br/>95<br/>93<br/>91<br/>89<br/>87<br/>85<br/>83<br/>81<br/>79<br/>77<br/>75<br/>73<br/>71<br/>69<br/>67<br/>65<br/>63<br/>61<br/>59<br/>57<br/>55<br/>53<br/>51<br/>49<br/>47<br/>45<br/>43<br/>41<br/>39<br/>37<br/>35<br/>33<br/>31<br/>29<br/>27<br/>25<br/>23<br/>21<br/>19<br/>17<br/>15<br/>13<br/>11<br/>9<br/>7<br/>5<br/>3<br/>==1==<br/>1 | 1<br/>3<br/>5<br/>7<br/>9<br/>11<br/>13<br/>15<br/>17<br/>19<br/>21<br/>23<br/>25<br/>27<br/>29<br/>31<br/>33<br/>35<br/>37<br/>39<br/>41<br/>43<br/>45<br/>47<br/>49<br/>51<br/>53<br/>55<br/>57<br/>59<br/>61<br/>63<br/>65<br/>67<br/>69<br/>71<br/>73<br/>75<br/>77<br/>79<br/>81<br/>83<br/>85<br/>87<br/>89<br/>91<br/>93<br/>95<br/>97<br/>99<br/>101<br/>103<br/>105<br/>107<br/>109<br/>111<br/>113<br/>115<br/>117<br/>119<br/>121<br/>123<br/>125<br/>127<br/>129<br/>131<br/>133<br/>135<br/>137<br/>139<br/>141<br/>143<br/>145<br/>147<br/>149<br/>151<br/>153<br/>155<br/>157<br/>159<br/>161<br/>163<br/>165<br/>167<br/>169<br/>171<br/>173<br/>175<br/>177<br/>179<br/>181<br/>183<br/>185<br/>187<br/>189<br/>191<br/>193<br/>195<br/>197<br/>==199==<br/>199 |

