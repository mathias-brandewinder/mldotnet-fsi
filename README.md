# mldotnet-fsi

F# scripting setup to use ML.NET in VS Code with dotnet core.

## Usage

0) Prerequisites 

- dotnet core 3.1 installed 
- VS Code with the Ionide plugin
- Ionide settings: `FSharp: Use Sdk Scripts` (Use 'dotnet fsi' instead of 'fsi.exe'/'fsharpi')

1) Run the following in the terminal to download the ML.NET packages:

```
dotnet tool restore
dotnet paket restore
```

2) The script `script.fsx` should be ready to run: select all and run `Alt+Enter` on Windows to execute.
