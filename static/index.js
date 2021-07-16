let fileInput = document.getElementById("file-upload-input");
let fileSelect = document.getElementsByClassName("file-upload-select")[0];
fileSelect.onclick = function() {
	fileInput.click();
}
fileInput.onchange = function() {
	let filename = fileInput.files[0].name;
	let selectName = document.getElementsByClassName("file-select-name")[0];
	selectName.innerText = filename;
}

document.getElementById('proceed').addEventListener('click',function(){
	document.getElementById('landing').style.display='none'
	document.getElementById('formContainer').style.display='block'
})

document.getElementById('isSelect').addEventListener('click',function(){
	if(document.getElementById('isSelect').checked){
		document.getElementById('uploadSource').placeholder="INPUT COLUMNS TO SELECT(comma separate)"
	}
	else{
		document.getElementById('uploadSource').placeholder="INPUT COLUMNS TO DROP(comma separate)"
	}
})