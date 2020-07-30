console.log("adentro");
const path = require('path')
const {spawn} = require('child_process')

let publicar = (arg1,req, res) =>{
return new Promise((resolve, reject)=>{

function runScript(){
   return spawn('python', [
      "-u",
      path.join(__dirname, 'ocr.py'),arg1
   ]);
}
const subprocess = runScript()


// print output of script
subprocess.stdout.on('data', (data) => {
   var dat=data
   console.log(`entrando al fin ${data}`)
   resolve(dat);
});
subprocess.stderr.on('data', (data) => {
   console.log(`error:${data}`);
   resolve(data);
});
subprocess.stderr.on('close', () => {
   var close="Closed"
   resolve(console.log(close));
   
});
})
};

module.exports = {
    publicar
}