# rsync -r source destination

# -r -> -a/--archive，递归同步 + 同步元信息(修改时间、权限等)。由于 rsync 默认使用文件大小和修改时间决定文件是否需要更新
# rsync -a source destination 

# 指定端口 -e 不能省略
# rsync -av -e 'ssh -p 2234' source/ user@remote_host:/destination

# 排除文件目录
# rsync -av --exclude={'file1.txt', 'dir1/', 'dir2/*'} source/ destination 
# rsync -av --exclude-from=exclude.list source/ destination 

# 增量备份
# rsync -a --link-dest /compare/path /source/path /target/path

# 镜像副本 --delete 将删除只存在于目标目录、不存在于源目录的文件
# rsync -av --delete source/ destination

# -v 参数表示输出细节 -vv 表示输出更详细的信息 -vvv 表示输出最详细的信息

rsync -av -e 'ssh -p 23' --link-dest ./ $TRAIN_SERVER:~/project/deeplabv3plus/ ./