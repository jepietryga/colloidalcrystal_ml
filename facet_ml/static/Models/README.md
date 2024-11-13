<h2> Downloading Files </h2>

<p>To download each file, go to the following resource or the associated command:</p>



<li> Models trained as part of the work on <a href="https://doi.org/10.5281/zenodo.14019586">Zenodo</a> </li>

<li>Segment Anything Vi-T Large <code>wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth </code></li>


</ul>

<h2> Small Notes </h2>
<p> It is noted that the Background Pixel Classifier model is quite large. This is due to the depth of the trees (averaging 57) in the RandomForestClassifer. For reproducibility, the model is shared as is, but it is recommended to focus on optimization of models in future work. </p>