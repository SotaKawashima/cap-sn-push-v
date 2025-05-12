use std::{
    borrow::Cow,
    ffi::OsStr,
    fs::File,
    io,
    marker::PhantomData,
    path::{Path, PathBuf},
    str::FromStr,
};

use itertools::{Itertools, ProcessResults};
use serde::{de::DeserializeOwned, Deserialize};

#[derive(Debug)]
pub struct MyPath<T>(PathBuf, PhantomData<T>);

impl<T> Clone for MyPath<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<'de, T> Deserialize<'de> for MyPath<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let path = PathBuf::deserialize(deserializer)?;
        Ok(Self(path, PhantomData))
    }
}

impl<T> From<PathBuf> for MyPath<T> {
    #[inline]
    fn from(value: PathBuf) -> Self {
        Self::new(value)
    }
}

impl<T> From<String> for MyPath<T> {
    #[inline]
    fn from(value: String) -> Self {
        Self::new(value.into())
    }
}

impl<T> FromStr for MyPath<T> {
    type Err = <PathBuf as FromStr>::Err;
    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        PathBuf::from_str(s).map(Self::new)
    }
}

impl<T, S: ?Sized + AsRef<OsStr>> From<&S> for MyPath<T> {
    #[inline]
    fn from(value: &S) -> Self {
        Self::new(value.into())
    }
}

impl<T> MyPath<T> {
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        self.0.to_string_lossy()
    }

    pub fn verified<P: AsRef<Path>>(&self, at: P) -> Result<PathBuf, io::Error> {
        let path = if !self.0.is_absolute() {
            at.as_ref().join(&self.0)
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

    fn new(buf: PathBuf) -> Self {
        Self(buf, PhantomData)
    }

    pub fn verified_child<P: AsRef<Path>>(&self, at: P) -> Result<PathBuf, io::Error> {
        Ok(self
            .verified(at)?
            .parent()
            .map_or_else(|| "./".into(), |p| p.to_path_buf()))
    }

    pub fn parse_at<P, F>(self, at: P, f: F) -> anyhow::Result<T>
    where
        P: AsRef<Path>,
        F: Fn(PathBuf) -> anyhow::Result<T>,
    {
        f(self.verified(at)?)
    }
}

// pub fn read_csv_and_then<T, P, F, U, E>(path: P, f: F) -> anyhow::Result<Vec<U>>
// where
//     T: DeserializeOwned,
//     P: AsRef<Path>,
//     F: Fn(T) -> Result<U, E>,
//     E: std::error::Error + Send + Sync + 'static,
// {
//     let file = File::open(path)?;
//     let mut rdr = csv::Reader::from_reader(file);
//     Ok(rdr
//         .deserialize::<T>()
//         .into_iter()
//         .process_results(|iter| iter.map(f).try_collect())??)
// }

pub fn read_csv_with<T, P, F, U>(path: P, processor: F) -> anyhow::Result<U>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
    F: FnOnce(ProcessResults<csv::DeserializeRecordsIter<File, T>, csv::Error>) -> U,
{
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    Ok(rdr.deserialize::<T>().process_results(processor)?)
}

pub fn read_csv<T, P>(path: P) -> anyhow::Result<Vec<T>>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
{
    read_csv_with(path, |iter| iter.collect_vec())
}

#[cfg(test)]
mod tests {
    use super::MyPath;

    #[test]
    fn test_my_path() {
        let p: MyPath<()> = "./hoge/fuga".into();
        let q = p.verified("");
        println!("{q:?}");
        assert!(q.is_err());
        let p: MyPath<()> = "./src/".into();
        assert!(p.verified("").is_ok());
        let p: MyPath<()> = "./src/io.rs".into();
        assert!(p.verified("").is_ok());
    }
}
