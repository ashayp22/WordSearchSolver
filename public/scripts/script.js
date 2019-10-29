function yo() {
  //show update on screen

  document.getElementById("addword").value = document.getElementById("allwords").innerHTML;

  document.getElementById("form").style.display = "none";
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

// Trigger photo take
function takePhoto() {
  var ctx = document.getElementById('canvas').getContext('2d');
	ctx.drawImage(video, 0, 0, 640, 480);
  var image = new Image();
  image.id = "pic";
  image.src = document.getElementById('canvas').toDataURL();

  document.getElementById("option2").src = image.src;
  document.getElementById("option2").value = image.src;

}
