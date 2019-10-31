
var wd = require("./index.js");

var options = { exact:false, hyperlinks: "none" };

wd.getDef("leprome", "fr", options, function(definition) {
	console.log(definition);
});