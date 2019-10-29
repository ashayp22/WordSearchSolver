function yo() {
  //show update on screen

  document.getElementById("addword").value = document.getElementById("allwords").innerHTML;

  document.getElementById("form").style.display = "none";
  document.getElementById("edit").style.display = "none";

  document.getElementById("loader").style.visibility = "visible";
  document.getElementById("loader").style.display = "inline-block";

  document.getElementById("loadingtext").style.visibility = "visible";
  document.getElementById("loadingtext").style.display = "inline-block";

  console.log("made changes");
}


var words = [];

function addWord() {
  var newWord = document.getElementById("addword").value;
  newWord = newWord.replace(/\s+/g, '');
  newWord = newWord.toLowerCase();
  if(document.getElementById("allwords").innerHTML.length != 0) {
    document.getElementById("allwords").innerHTML = document.getElementById("allwords").innerHTML + "," + newWord;
    words.push(newWord);
  } else {
    document.getElementById("allwords").innerHTML = newWord;
    words.push(newWord);
  }

}

function removeWord() {
  if(document.getElementById("allwords").innerHTML.length != 0) {

    var s = document.getElementById("allwords").innerHTML;

    s = s.replace(words.pop(), ""); //remove last word

    s = s.substring(0, s.length-1); //removes comma

    console.log(s);
    console.log(words);

    document.getElementById("allwords").innerHTML = s;
  }
}

function clearWords() {
  document.getElementById("allwords").innerHTML = "";
}


//camera

// Grab elements, create settings, etc.
var video;

function setup() {
  video = document.getElementById('video');
  // Elements for taking the snapshot

  // Get access to the camera!
  if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      // Not adding `{ audio: true }` since we only want video now
      navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
          //video.src = window.URL.createObjectURL(stream);
          video.srcObject = stream;
          video.play();
      });
  }
}


var points = [];
var mousedown = false;
var evt;
var currentImage;

// Trigger photo take
function takePhoto() {
  var canvas = document.getElementById('canvas');
  var ctx = canvas.getContext('2d');
  currentImage = video;
	ctx.drawImage(video, 0, 0, 640, 480);

  cancelAnimationFrame(animate);
  mousedown = false;
  points = [];

  points.push(new Circle(10, 10)); //top left
  points.push(new Circle(10, 465)); //bottom left
  points.push(new Circle(625, 10)); //top right
  points.push(new Circle(625, 465)); //bottom right


  document.getElementById("cropInfo").style.visibility = "visible";
  document.getElementById("cropInfo").style.display = "inline-block";

  document.getElementById("save").style.visibility = "visible";
  document.getElementById("save").style.display = "inline-block";

  animate();

  // var image = new Image();
  // image.id = "pic";
  // image.src = canvas.toDataURL();

  // document.getElementById("option2").src = image.src;

}

function Circle(x, y) {
  this.x = x;
  this.y = y;
}

function getMousePos(canvas, evt) {
    var rect = canvas.getBoundingClientRect();
    return {
      x: evt.pageX - rect.left,
      y: evt.pageY - rect.top
    };
}

function animate() {

  //animate
  var c = document.getElementById("canvas");
  var ctx = c.getContext("2d");

  //clear board

  ctx.drawImage(currentImage, 0, 0, 640, 480);

//move circles

  if(mousedown) {
    var position = getMousePos(c, evt);

    for(i = 0; i < 4; i++) {
      if(Math.abs(position.x - points[i].x) < 15 && Math.abs(position.y - points[i].y) < 15) {
        points[i].x = position.x;
        points[i].y = position.y;

        //change the others to make it align

        if(i == 0) {
          points[1].x = position.x;
          points[2].y = position.y;
        } else if(i ==1) {
          points[0].x = position.x;
          points[3].y = position.y;
        } else if(i == 2) {
          points[3].x = position.x;
          points[0].y = position.y;
        } else if(i == 3) {
          points[2].x = position.x;
          points[1].y = position.y;
        }

        break;

      }
    }

  }

  //draw circles

    for(i = 0; i < 4; i++) {
      ctx.beginPath();
      ctx.arc(points[i].x, points[i].y, 10, 0, 2 * Math.PI);
      ctx.fillStyle = "#2460a4";
      ctx.fill();
    }

    //draw lines

    ctx.setLineDash([5, 3]);

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    ctx.lineTo(points[1].x, points[1].y);
    ctx.stroke();

    ctx.beginPath();

    ctx.moveTo(points[1].x, points[1].y);
    ctx.lineTo(points[3].x, points[3].y);
    ctx.stroke();

    ctx.beginPath();


    ctx.moveTo(points[3].x, points[3].y);
    ctx.lineTo(points[2].x, points[2].y);
    ctx.stroke();

    ctx.beginPath();

    ctx.moveTo(points[2].x, points[2].y);
    ctx.lineTo(points[0].x, points[0].y);
    ctx.stroke();

  requestAnimationFrame(animate);
}

$(document).ready(function() {
  $('#canvas').mousedown(function () {
    mousedown = true;
  });

  $('#canvas').mousemove(function (event) {
    evt = event;
  });

  $('#canvas').mouseup(function () {
      mousedown = false;
  });
});


function saveImage() {

}
