catalog_3d <- function(data, time.begin=NULL, study.start=NULL,
                       study.end=NULL, study.length=NULL,
                       lat.range=NULL, long.range=NULL, depth.range=NULL, # Added depth.range
                       region.poly=NULL, mag.threshold=NULL,
                       flatmap=TRUE, dist.unit="degree", 
                       roundoff=TRUE, tz="GMT")
{
  data <- as.data.frame(data)
  dnames <- tolower(names(data))
  names(data) <- dnames
  
  # --- CHANGE 1: Add 'depth' to required variable names ---
  vnames <- c("date", "time", "long", "lat", "mag", "depth") 
  
  if (!all(vnames %in% dnames))
    stop(paste("argument", sQuote("data"),
               "must be a data frame with column names ",
               toString(sQuote(vnames))))
  
  if (any(is.na(data[, vnames])))
    stop(paste(sQuote(vnames), "must not contain NA values"))
  
  # --- CHANGE 2: Check depth is numeric ---
  if (!is.numeric(data$lat) || !is.numeric(data$long) || 
      !is.numeric(data$mag) || !is.numeric(data$depth))
    stop("lat, long, mag, and depth columns must be numeric vectors")

  # extract spatial coordinates, magnitude, and depth
  xx <- data$long  
  yy <- data$lat   
  zz <- data$depth # --- CHANGE 3: Extract Depth ---
  mm <- data$mag   
  
  # accounting for round-off error
  if(roundoff)
  {
    xx <- roundoffErr(xx)
    yy <- roundoffErr(yy)
    # Depth usually doesn't need roundoffErr unless it's very discrete, 
    # but we can skip it for now or add it if your data is grid-based.
  }

  # extract date and time (No changes here)
  dt <- as.POSIXlt.character(paste(data$date, data$time), tz=tz)
  if (sum(duplicated(dt)) > 0)
  {
    dtidx <- which(diff(dt) == 0)
    for (i in dtidx)
    {
      dt[i + 1] <- dt[i] + as.difftime(1, units="secs")
    }
    warning(paste("more than one event has occurred simultaneously!",
                  "\ncheck events", toString(dtidx),
                  "\nduplicated times have been altered by one second"))
  }
  if (is.unsorted(dt))
  {
    warning(paste("events were not chronologically sorted:",
                  "they have been sorted in ascending order"))
    # Re-order ALL vectors including depth
    ord <- order(dt)
    data <- data[ord, ]
    xx <- xx[ord]
    yy <- yy[ord]
    zz <- zz[ord] # Sort depth
    mm <- mm[ord]
    dt <- sort(dt)
  }

  # Time filtering logic (No changes here)
  if (is.null(time.begin)) time.begin <- min(dt)
  else {
    time.begin <- as.POSIXlt(time.begin, tz=tz)
    if (all(dt < time.begin)) stop(paste("change time.begin: no event after:", sQuote(time.begin)))
  }
  if (is.null(study.start)) study.start <- time.begin
  else {
    study.start <- as.POSIXlt(study.start, tz=tz)
    if (study.start < time.begin) stop(paste("study.start < time.begin"))
  }
  if (!is.null(study.length)) {
    if (!is.null(study.end)) stop("either study.end or study.length, not both")
    else {
      if(!is.numeric(study.length) || length(study.length) > 1) stop("study.length must be single numeric")
      study.end <- study.start + study.length * 24 * 60 * 60
    }
  }
  if (is.null(study.end)) study.end <- max(dt)
  else {
    study.end <- as.POSIXlt(study.end, tz=tz)
    if (study.end < study.start) stop("study.end < study.start")
  }
  tt <- date2day(dt, time.begin, tz=tz)

  # Spatial region (Lat/Long)
  if (is.null(lat.range)) lat.range <- range(yy) + 0.01 * diff(range(yy)) * c(-1, 1)
  if (is.null(long.range)) long.range <- range(xx) + 0.01 * diff(range(xx)) * c(-1, 1)

  # --- CHANGE 4: Depth Range Filtering ---
  if (is.null(depth.range)) {
    depth.range <- range(zz) # Default to min/max of data
  } else if (!is.vector(depth.range) || length(depth.range) != 2) {
    stop("depth.range must be a vector of length 2 giving (depth.min, depth.max)")
  }

  # Region Polygon Logic (2D boundary only - standard practice)
  if (is.null(region.poly))
  {
    region.poly <- list(long=c(long.range, rev(long.range)),
                        lat=rep(lat.range, each=2))
    region.win <- spatstat.geom::owin(xrange=long.range, yrange=lat.range)
  } else {
    # (Same validation logic as original...)
    if (is.data.frame(region.poly)) region.poly <- as.list(region.poly)
    if (!is.list(region.poly) || !all(c("lat", "long") %in% names(region.poly)))
      stop("region.poly must be a list with components lat and long")
    region.win <- spatstat.geom::owin(poly=list(x=region.poly$long, y=region.poly$lat))
  }

  # Magnitude threshold
  if (is.null(mag.threshold)) mag.threshold <- min(mm)

  # Project coordinates
  longlat.coord <- data.frame(long=xx, lat=yy, depth=zz) # Add depth to backup
  if (flatmap)
  {
    # We only project X and Y. Z (depth) is usually already in KM.
    proj <- longlat2xy(long=xx, lat=yy, region.poly=region.poly,
                       dist.unit=dist.unit)
    xx <- proj$x
    yy <- proj$y
    region.win <- proj$region.win
  }

  # --- CHANGE 5: Filter logic including Depth ---
  ok <- (dt <= study.end) & (dt >= time.begin) & (mm >= mag.threshold) & 
        (zz >= depth.range[1]) & (zz <= depth.range[2])
  
  xx <- xx[ok]
  yy <- yy[ok]
  zz <- zz[ok] # Filter depth
  tt <- tt[ok]
  mm <- mm[ok] - mag.threshold
  
  # Determine if inside 2D polygon region
  flag <- as.integer(spatstat.geom::inside.owin(xx, yy, region.win))
  flag[dt[ok] < study.start] <- -2
  
  # --- CHANGE 6: Update revents matrix to include Z ---
  # Structure: t, x, y, z, mag, flag, ...
  revents <- cbind(tt, xx, yy, zz, mm, flag, bkgd=0, prob=1, lambd=0)
  
  longlat.coord <- longlat.coord[ok, ]
  longlat.coord$flag <- flag
  longlat.coord$dt <- dt[ok]
  
  # --- CHANGE 7: Update spatstat object (ppx instead of ppp) ---
  # 'ppx' allows multidimensional point patterns. 
  # We mark 'z' as a spatial coordinate ("s").
  X <- spatstat.geom::ppx(data.frame(t=tt, x=xx, y=yy, z=zz, m=mm),
                       coord.type=c("t", "s", "s", "s", "m"))

  # Polygon extraction (same as before)
  switch(region.win$type, polygonal= {
    px <- region.win$bdry[[1]]$x
    py <- region.win$bdry[[1]]$y
  }, rectangle = {
    px <- c(region.win$xrange, rev(region.win$xrange))
    py <- rep(region.win$yrange, each=2)
  })
  np <- length(px) + 1
  px[np] <- px[1]
  py[np] <- py[1]
  rpoly <- cbind(px, py)

  rtperiod <- c(date2day(study.start, time.begin, tz=tz),
                date2day(study.end, time.begin, tz=tz))
  
  # --- CHANGE 8: Output list includes depth info ---
  out <- list(revents=revents, rpoly=rpoly, rtperiod=rtperiod, X=X,
              region.poly=region.poly, region.win=region.win,
              time.begin=time.begin, study.start=study.start,
              study.end=study.end, study.length=study.length,
              mag.threshold=mag.threshold, 
              depth.range=depth.range, # Save the depth range used
              longlat.coord=longlat.coord,
              dist.unit=dist.unit)
  
  class(out) <- "catalog" # You might want to name this "catalog3d" to avoid confusion
  return(out)
}

# --- Updated Print Method ---
print.catalog <- function (x, ...)
{
  cat("3D earthquake catalog:\n  time begin", as.character(x$time.begin),
      "\n  study period:", as.character(x$study.start),
      " to ", as.character(x$study.end), "(T =", diff(x$rtperiod), "days)")
  cat("\ngeographical region:\n  ")
  switch(x$region.win$type, rectangle={
    cat("  rectangular = [", x$region.poly$long[1], ",", x$region.poly$long[2],
        "] x [", x$region.poly$lat[1], x$region.poly$lat[2], "]\n")
  }, polygonal={
    cat("  polygonal with vertices:\n")
    print(cbind(lat=x$region.poly$lat, long=x$region.poly$long))
  })
  
  # --- Add Depth Info to Print ---
  if(!is.null(x$depth.range)) {
    cat("  depth range:", x$depth.range[1], "to", x$depth.range[2], "km\n")
  }
  
  cat("threshold magnitude:", x$mag.threshold)
  cat("\nnumber of events:\n  total events", nrow(x$revents),
      ":", sum(x$revents[, 6] == 1), "target events, ", # Note index change to 6
      sum(x$revents[, 6] != 1), "complementary events\n  (",
      sum(x$revents[, 6] == 0), "events outside geographical region,",
      sum(x$revents[, 6] == -2), "events outside study period)\n")
}

plot.catalog <- function(x, ...)
{
  oldpar <- par(no.readonly = TRUE)
  lymat <- matrix(c(1, 1, 2, 1, 1, 3, 4, 5, 6), 3, 3)
  layout(lymat)
  par(mar=c(4, 4.25, 1, 1))
  plot(x$longlat.coord$long, x$longlat.coord$lat, xlab="long", ylab="lat",
       col=8, cex=2 * (x$revents[, 4] + 0.1)/max(x$revents[, 4]),
       asp=TRUE, axes=FALSE)
  maps::map('world', add=TRUE, col="grey50")
  axis(1); axis(2)
  ok <- x$revents[, 6] == 1
  points(x$longlat.coord$long[ok], x$longlat.coord$lat[ok], col=4,
         cex=2 * (x$revents[ok, 4] + 0.1)/max(x$revents[ok, 4]))
  polygon(x$region.poly$long, x$region.poly$lat, border=2)
  #
  mbk <- seq(0, ceiling(10 * max(x$revents[, 4])) / 10 + 1e-06, 0.1) + x$mag.threshold
  mct <- cut(x$revents[, 4] + x$mag.threshold, mbk, include.lowest=TRUE)
  mcc <- log10(rev(cumsum(rev(table(mct)))))
  valid <- is.finite(mcc)
  mcc <- mcc[valid]
  mbk <- mbk[valid]
  plot(mbk[-length(mbk)], mcc, type="b",
       xlab="mag", ylab=expression(log[10]*N[mag]), axes=FALSE)
  graphics::abline(stats::lm(mcc ~ mbk[-length(mbk)]), col=4, lty=4)
  graphics::axis(1); graphics::axis(2)
  #
  tbk <- seq(0, max(x$revents[, 1]), l=100)
  tct <- cut(x$revents[, 1], tbk)
  tcc <- (cumsum(table(tct)))
  plot(tbk[-length(tbk)], tcc, type="l",
       xlab="time", ylab=expression(N[t]), axes=FALSE)
  tok <- (tbk[-length(tbk)] >= x$rtperiod[1]) & (tbk[-length(tbk)] <= x$rtperiod[2])
  graphics::abline(stats::lm(tcc[tok] ~ tbk[-length(tbk)][tok]), col=4, lty=4)
  graphics::axis(1); graphics::axis(2)
  abline(v=x$rtperiod[1], col=2, lty=2)
  abline(v=x$rtperiod[2], col=2, lty=2)
  #
  plot(x$revents[, 1], x$revents[, 3], xlab="time", ylab="lat",
       cex=2 * (x$revents[, 4] + 0.1)/max(x$revents[, 4]), col=8, axes=FALSE)
  points(x$revents[ok, 1], x$revents[ok, 3], col=4,
         cex=2 * (x$revents[ok, 4] + 0.1)/max(x$revents[ok, 4]))
  axis(1); axis(2)
  #
  plot(x$revents[, 1], x$longlat.coord$long, xlab="time", ylab="long",
       cex=2 * (x$revents[, 4] + 0.1)/max(x$revents[, 4]), col=8, axes=FALSE)
  points(x$revents[ok, 1], x$longlat.coord$long[ok], col=4,
         cex=2 * (x$revents[ok, 4] + 0.1)/max(x$revents[ok, 4]))
  axis(1); axis(2)
  #
  plot(x$revents[, 1], x$revents[, 4] + x$mag.threshold, xlab="time",
       ylab="mag", pch=20, cex=0.5, col=8, axes=FALSE)
  points(x$revents[ok, 1], x$revents[ok, 4] + x$mag.threshold, col=4, pch=20, cex=0.5)
  axis(1); axis(2)
  #
  layout(1)
  par(oldpar)
}
