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
