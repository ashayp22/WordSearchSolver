const PORT = process.env.PORT || 3000; //either heroku port or local port
const express = require('express');
const bodyParser = require('body-parser');
const app = express()
const formidable = require('formidable');
const fs = require('fs');
const pngToJpeg = require('png-to-jpeg');
const path = require('path')
const Jimp = require('jimp');
const exec = require('child_process').exec;
const mv = require('mv');


app.use(express.static('public'));
app.use(bodyParser.urlencoded({ extended: true }));
app.set('view engine', 'ejs')

app.get('/', function (req, res) { //handles get request
  res.render('index');
})

app.post('/fileupload', function (req, res) { //handles post request
  var form = new formidable.IncomingForm();
    form.parse(req, function (err, fields, files) {

      //parsing words rows and columns

      var rows = fields.rows;
      var columns = fields.columns;

      var w = fields.words;

      //check to make sure the right data was sent
      if(files.filetoupload.name == "" || rows <= 0 || columns <= 0 || w.length == 0) {
        res.render('index');
        res.end(); //end
        return;
      }

      rows = rows + "";
      columns = columns + "";

      var obj = {
       board: {},
       words: {}
     };

     obj.board = {rows: rows, columns: columns};
     obj.words = {list: w};

     var json = JSON.stringify(obj);
     fs.writeFile('public/resources/info.json', json, 'utf8', writeCallback);

      //parsing image


      var extension = path.extname(files.filetoupload.name);

      //PNG
      if(extension == '.png') {
        Jimp.read(files.filetoupload.path)
          .then(image => {
            return image
              .quality(100) // set JPEG quality
              .greyscale() // set greyscale
              .write('public/resources/answer.jpg'); // save
          })
          .catch(err => {
            console.error(err);
          });
          //now we add in the highlights

          var cmd = "py -3.6 findwords.py";
          exec(cmd, allDone);

          function allDone() {
            console.log("called python")
            res.render('response');
            res.end(); //end
          }
      } else if(extension == ".jpg"){ //JPEG

        mv(files.filetoupload.path, "public/resources/answer.jpg", function(err) {
          // done. it tried fs.rename first, and then falls back to
          // piping the source file to the dest file and then unlinking
          // the source file.
          if (err) throw err;
          console.log('Rename complete!');

          //now we add in the highlights

          var cmd = "py -3.6 findwords.py";
          exec(cmd, allDone);

          function allDone() {
            console.log("called python")
            res.render('response');
            res.end(); //end
          }
        });

        // fs.rename(files.filetoupload.path, "public/resources/answer.jpg", function (err) {
        //
        // });
      } else { //wrong extension
        console.log('wrong extension!');
        res.render('index');
        res.end(); //end
      }
  })
})

app.listen(PORT, function () {
  console.log('go to http://localhost:3000/')
})

function writeCallback(err) { //callback for the function above
  console.log("updated node file");
}
