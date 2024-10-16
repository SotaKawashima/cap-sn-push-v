use std::{
    borrow::Cow,
    ffi::OsStr,
    io,
    path::{Path, PathBuf},
    str::FromStr,
};

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct MyPath(PathBuf);

impl From<String> for MyPath {
    #[inline]
    fn from(value: String) -> Self {
        Self(value.into())
    }
}

impl FromStr for MyPath {
    type Err = <PathBuf as FromStr>::Err;
    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        PathBuf::from_str(s).map(Self)
    }
}

impl<T: ?Sized + AsRef<OsStr>> From<&T> for MyPath {
    #[inline]
    fn from(value: &T) -> Self {
        Self(value.into())
    }
}

impl MyPath {
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        self.0.to_string_lossy()
    }

    pub fn verified<P: AsRef<Path>>(&self, root: P) -> Result<PathBuf, io::Error> {
        let path = if !self.0.is_absolute() {
            root.as_ref().join(&self.0)
        } else {
            self.0.clone()
        };
        if path.try_exists()? {
            Ok(path)
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Path {} does not exist.", path.to_string_lossy()),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MyPath;

    #[test]
    fn test_my_path() {
        let p: MyPath = "./hoge/fuga".into();
        let q = p.verified("");
        println!("{q:?}");
        assert!(q.is_err());
        let p: MyPath = "./src/".into();
        assert!(p.verified("").is_ok());
        let p: MyPath = "./src/io.rs".into();
        assert!(p.verified("").is_ok());
    }
}
