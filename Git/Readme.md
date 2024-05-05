# Git Bash Commands

- [For more commands](https://dzone.com/articles/top-20-git-commands-with-examples),[and this ](https://opensource.com/article/18/2/how-clone-modify-add-delete-git-files)
- [Git Docs](https://git-scm.com/doc)
- [Git tutorials from official doc](https://git-scm.com/docs/gittutorial)
- [Undo the things of 1 file after commited](https://stackoverflow.com/questions/692246/undo-working-copy-modifications-of-one-file-in-git)
- [Git from scratch Article](https://medium.com/@jake.page91/the-guide-to-git-i-never-had-a89048d4703a)
## Clone the Repository

To clone the GitHub repository where you want to create the README file, use the following command:

```bash
git clone <repository URL>
```
## Create folder
```
mkdir <folder name>
```
## Change Directory to the Repository
```
cd <repository directory>
```
## Create the README File
You can create the README file using a text editor or use the touch command to create an empty file. To create a README.md file, you can use the following command:
```
touch README.md
```

## Edit the README File
Use a text editor to edit the README.md file. You can open it with a text editor like nano or vim or echo. For example:
```
nano README.md
```

## Save and Exit the Text Editor
After editing the README file, save your changes and exit the text editor. For example, in nano, you can press Ctrl + O to save and Ctrl + X to exit.

## Commit Your Changes
Add the README.md file to the staging area and commit it to your local Git repository:
```
git add README.md
git commit -m "Add README with Git Bash commands"
```
## Push to GitHub
Push your changes to your GitHub repository:
```
git push
```

# Working with Your GitHub Repository
## Create a New Branch
To create a new branch in your Git repository, use the following command:
```
git branch <branch-name>
```

## Switch to a Different Branch
You can switch to a different branch using the checkout command:
```
git checkout <branch-name>
```

## Make Changes to Files
Edit or create new files in your repository. You can use a text editor or command-line text editing tools.

## View Changes
To see the changes you've made to your files, use the following command:
```
git status
```
This command provides information about the files that have been modified or added.

## Stage Changes
Stage changes for a commit using the add command. For example, to stage all changes, use:

```
git add .
```
To stage specific files, replace . with the file names.

## Commit Changes
Commit your staged changes with a meaningful message:

```
git commit -m "Your commit message here"
```
## Push Changes to GitHub
Push your committed changes to your GitHub repository:

```
git push
```
## Pull Changes from GitHub
To update your local repository with changes from the remote repository, use:
```
git pull
```
## Merge Branches
If you have multiple branches and want to merge changes from one branch into another, use the merge command:

```
git merge <branch-name>
```
Replace <branch-name> with the name of the branch you want to merge.

##  TO view more about commit details
```
git log --stat

```
## TO check the differences made
```
git diff
```

##  To view more commit details
```
git log 

```
## Resolve Merge Conflicts
If there are merge conflicts during the branch merge, resolve them in your text editor and then commit the resolved changes.

## To add a repo
```
git init <repo0 name >
```

## Configure email
```
git config --global user.email "your.email@domain.com"
```
## Configure name 
```
git config -- global user.name "Your name"
```
## Making a commit
```
git commit -m "Commit message here"
```

