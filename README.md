<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  
</head>
<body>
 <video width="560" height="315" controls>
    <source src="https://www.vecteezy.com/free-videos/reading-cartoon" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <h1 style="font-size: 36px;">Literary-Alchemy</h1>
  
  <hr><p>
    <ul>
    <li>'Literary-Alchemy' is a python powered hybrid book recommender system integrating collaborative filtering and content-based filtering.</li>  
    <li>Implemented the NearestNeighbors model from scikit-learn with Minkowski distance  and brute-force algorithm for similarity computations.</li>
    <li>Integrated collaborative filtering methods to recommend books based on user preferences and content-based filtering to suggest books similar to those already read.</li>
    <li>Leveraged deep learning techniques to enhance recommendation accuracy and provide personalized book suggestions.</li>
    <li>Utilized Python's scikit-learn library for implementing the NearestNeighbors model and handling similarity computations.</li>
  </ul>
</p>
  <br>
  <h2 style="font-size: 36px;">Input</h2><hr>
  <p>
     <ul>
       <li>User_id</li>
       <li>Book read by user earlier</li>
       <li>Weight </li></ul>
     </ul></p>
   
   <h2 style="font-size: 36px;">Output</h2><hr>
   <br>
  
  <img src="https://github.com/ayushshauryajha/Literary_Alchemy/blob/main/Requirements/hr1.png" alt="Dataset Photo">
 <br>
 </body>
 </html>
  <h2 style="font-size: 36px;">About the model</h2><hr>
   <p>
     Model outputs the books using both User_id and books raed by the user earlier by taking into account both models (1)User based collaborative filtering and (2)Item based collaborative filtering. The number of books that are recommended on the basis of user_id and that on the basis of books read earlier depends on the weigth provided by the user .
 </p>
